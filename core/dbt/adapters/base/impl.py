import abc
from contextlib import contextmanager

import agate
import pytz
import six

import dbt.exceptions
import dbt.flags
import dbt.clients.agate_helper

from dbt.compat import abstractclassmethod, classmethod
from dbt.node_types import NodeType
from dbt.loader import GraphLoader
from dbt.logger import GLOBAL_LOGGER as logger
from dbt.utils import filter_null_values


from dbt.adapters.base.meta import AdapterMeta, available
from dbt.adapters.base import BaseRelation
from dbt.adapters.base import Column
from dbt.adapters.cache import RelationsCache


GET_CATALOG_MACRO_NAME = 'get_catalog'
FRESHNESS_MACRO_NAME = 'collect_freshness'


def _expect_row_value(key, row):
    if key not in row.keys():
        raise dbt.exceptions.InternalException(
            'Got a row without "{}" column, columns: {}'
            .format(key, row.keys())
        )
    return row[key]


def _relations_filter_schemas(schemas):
    def test(row):
        referenced_schema = _expect_row_value('referenced_schema', row)
        dependent_schema = _expect_row_value('dependent_schema', row)
        # handle the null schema
        if referenced_schema is not None:
            referenced_schema = referenced_schema.lower()
        if dependent_schema is not None:
            dependent_schema = dependent_schema.lower()
        return referenced_schema in schemas or dependent_schema in schemas
    return test


def _catalog_filter_schemas(manifest):
    """Return a function that takes a row and decides if the row should be
    included in the catalog output.
    """
    schemas = frozenset((d.lower(), s.lower())
                        for d, s in manifest.get_used_schemas())

    def test(row):
        table_database = _expect_row_value('table_database', row)
        table_schema = _expect_row_value('table_schema', row)
        # the schema may be present but None, which is not an error and should
        # be filtered out
        if table_schema is None:
            return False
        return (table_database.lower(), table_schema.lower()) in schemas
    return test


def _utc(dt, source, field_name):
    """If dt has a timezone, return a new datetime that's in UTC. Otherwise,
    assume the datetime is already for UTC and add the timezone.
    """
    if dt is None:
        raise dbt.exceptions.raise_database_error(
            "Expected a non-null value when querying field '{}' of table "
            " {} but received value 'null' instead".format(
                field_name,
                source))

    elif not hasattr(dt, 'tzinfo'):
        raise dbt.exceptions.raise_database_error(
            "Expected a timestamp value when querying field '{}' of table "
            "{} but received value of type '{}' instead".format(
                field_name,
                source,
                type(dt).__name__))

    elif dt.tzinfo:
        return dt.astimezone(pytz.UTC)
    else:
        return dt.replace(tzinfo=pytz.UTC)


class SchemaSearchMap(dict):
    """A utility class to keep track of what information_schema tables to
    search for what schemas
    """
    def add(self, relation):
        key = relation.information_schema_only()
        if key not in self:
            self[key] = set()
        self[key].add(relation.schema.lower())

    def search(self):
        for information_schema_name, schemas in self.items():
            for schema in schemas:
                yield information_schema_name, schema

    def schemas_searched(self):
        result = set()
        for information_schema_name, schemas in self.items():
            result.update(
                (information_schema_name.database, schema)
                for schema in schemas
            )
        return result

    def flatten(self):
        new = self.__class__()

        database = None
        # iterate once to look for a database name
        seen = {r.database.lower() for r in self if r.database}
        if len(seen) > 1:
            dbt.exceptions.raise_compiler_error(str(seen))
        elif len(seen) == 1:
            database = list(seen)[0]

        for information_schema_name, schema in self.search():
            new.add(information_schema_name.incorporate(
                path={'database': database, 'schema': schema},
                quote_policy={'database': False},
                include_policy={'database': False},
            ))

        return new


@six.add_metaclass(AdapterMeta)
class BaseAdapter(object):
    """The BaseAdapter provides an abstract base class for adapters.

    Adapters must implement the following methods and macros. Some of the
    methods can be safely overridden as a noop, where it makes sense
    (transactions on databases that don't support them, for instance). Those
    methods are marked with a (passable) in their docstrings. Check docstrings
    for type information, etc.

    To implement a macro, implement "${adapter_type}__${macro_name}". in the
    adapter's internal project.

    Methods:
        - exception_handler
        - date_function
        - list_schemas
        - drop_relation
        - truncate_relation
        - rename_relation
        - get_columns_in_relation
        - expand_column_types
        - list_relations_without_caching
        - is_cancelable
        - create_schema
        - drop_schema
        - quote
        - convert_text_type
        - convert_number_type
        - convert_boolean_type
        - convert_datetime_type
        - convert_date_type
        - convert_time_type

    Macros:
        - get_catalog
    """
    requires = {}

    Relation = BaseRelation
    Column = Column
    # This should be an implementation of BaseConnectionManager
    ConnectionManager = None

    # A set of clobber config fields accepted by this adapter
    # for use in materializations
    AdapterSpecificConfigs = frozenset()

    def __init__(self, config):
        self.config = config
        self.cache = RelationsCache()
        self.connections = self.ConnectionManager(config)
        self._internal_manifest_lazy = None

    ###
    # Methods that pass through to the connection manager
    ###
    def acquire_connection(self, name=None):
        return self.connections.set_connection_name(name)

    def release_connection(self):
        return self.connections.release()

    def cleanup_connections(self):
        return self.connections.cleanup_all()

    def clear_transaction(self):
        self.connections.clear_transaction()

    def commit_if_has_connection(self):
        return self.connections.commit_if_has_connection()

    def nice_connection_name(self):
        conn = self.connections.get_thread_connection()
        if conn is None or conn.name is None:
            return '<None>'
        return conn.name

    @contextmanager
    def connection_named(self, name):
        try:
            yield self.acquire_connection(name)
        finally:
            self.release_connection()

    @available.parse(lambda *a, **k: ('', dbt.clients.agate_helper()))
    def execute(self, sql, auto_begin=False, fetch=False):
        """Execute the given SQL. This is a thin wrapper around
        ConnectionManager.execute.

        :param str sql: The sql to execute.
        :param bool auto_begin: If set, and dbt is not currently inside a
            transaction, automatically begin one.
        :param bool fetch: If set, fetch results.
        :return: A tuple of the status and the results (empty if fetch=False).
        :rtype: Tuple[str, agate.Table]
        """
        return self.connections.execute(
            sql=sql,
            auto_begin=auto_begin,
            fetch=fetch
        )

    ###
    # Methods that should never be overridden
    ###
    @classmethod
    def type(cls):
        """Get the type of this adapter. Types must be class-unique and
        consistent.

        :return: The type name
        :rtype: str
        """
        return cls.ConnectionManager.TYPE

    @property
    def _internal_manifest(self):
        if self._internal_manifest_lazy is None:
            manifest = GraphLoader.load_internal(self.config)
            self._internal_manifest_lazy = manifest
        return self._internal_manifest_lazy

    def check_internal_manifest(self):
        """Return the internal manifest (used for executing macros) if it's
        been initialized, otherwise return None.
        """
        return self._internal_manifest_lazy

    ###
    # Caching methods
    ###
    def _schema_is_cached(self, database, schema):
        """Check if the schema is cached, and by default logs if it is not."""

        if dbt.flags.USE_CACHE is False:
            return False
        elif (database, schema) not in self.cache:
            logger.debug(
                'On "{}": cache miss for schema "{}.{}", this is inefficient'
                .format(self.nice_connection_name(), database, schema)
            )
            return False
        else:
            return True

    @classmethod
    def _relations_filter_table(cls, table, schemas):
        """Filter the table as appropriate for relations table entries.
        Subclasses can override this to change filtering rules on a per-adapter
        basis.
        """
        return table.where(_relations_filter_schemas(schemas))

    def _get_cache_schemas(self, manifest, exec_only=False):
        """Get a mapping of each node's "information_schema" relations to a
        set of all schemas expected in that information_schema.

        There may be keys that are technically duplicates on the database side,
        for example all of '"foo", 'foo', '"FOO"' and 'FOO' could coexist as
        databases, and values could overlap as appropriate. All values are
        lowercase strings.
        """
        info_schema_name_map = SchemaSearchMap()
        for node in manifest.nodes.values():
            if exec_only and node.resource_type not in NodeType.executable():
                continue
            relation = self.Relation.create_from(self.config, node)
            info_schema_name_map.add(relation)
        # result is a map whose keys are information_schema Relations without
        # identifiers that have appropriate database prefixes, and whose values
        # are sets of lowercase schema names that are valid members of those
        # schemas
        return info_schema_name_map

    def _relations_cache_for_schemas(self, manifest):
        """Populate the relations cache for the given schemas. Returns an
        iteratble of the schemas populated, as strings.
        """
        if not dbt.flags.USE_CACHE:
            return

        info_schema_name_map = self._get_cache_schemas(manifest,
                                                       exec_only=True)
        for db, schema in info_schema_name_map.search():
            for relation in self.list_relations_without_caching(db, schema):
                self.cache.add(relation)

        # it's possible that there were no relations in some schemas. We want
        # to insert the schemas we query into the cache's `.schemas` attribute
        # so we can check it later
        self.cache.update_schemas(info_schema_name_map.schemas_searched())

    def set_relations_cache(self, manifest, clear=False):
        """Run a query that gets a populated cache of the relations in the
        database and set the cache on this adapter.
        """
        if not dbt.flags.USE_CACHE:
            return

        with self.cache.lock:
            if clear:
                self.cache.clear()
            self._relations_cache_for_schemas(manifest)

    def cache_new_relation(self, relation):
        """Cache a new relation in dbt. It will show up in `list relations`."""
        if relation is None:
            name = self.nice_connection_name()
            dbt.exceptions.raise_compiler_error(
                'Attempted to cache a null relation for {}'.format(name)
            )
        if dbt.flags.USE_CACHE:
            self.cache.add(relation)
        # so jinja doesn't render things
        return ''

    ###
    # Abstract methods for database-specific values, attributes, and types
    ###
    @abstractclassmethod
    def date_function(cls):
        """Get the date function used by this adapter's database.

        :return: The date function
        :rtype: str
        """
        raise dbt.exceptions.NotImplementedException(
            '`date_function` is not implemented for this adapter!')

    @abstractclassmethod
    def is_cancelable(cls):
        raise dbt.exceptions.NotImplementedException(
            '`is_cancelable` is not implemented for this adapter!'
        )

    ###
    # Abstract methods about schemas
    ###
    @abc.abstractmethod
    def list_schemas(self, database):
        """Get a list of existing schemas.

        :param str database: The name of the database to list under.
        :return: All schemas that currently exist in the database
        :rtype: List[str]
        """
        raise dbt.exceptions.NotImplementedException(
            '`list_schemas` is not implemented for this adapter!'
        )

    @available.parse(lambda *a, **k: False)
    def check_schema_exists(self, database, schema):
        """Check if a schema exists.

        The default implementation of this is potentially unnecessarily slow,
        and adapters should implement it if there is an optimized path (and
        there probably is)
        """
        search = (
            s.lower() for s in
            self.list_schemas(database=database)
        )
        return schema.lower() in search

    ###
    # Abstract methods about relations
    ###
    @abc.abstractmethod
    @available.parse_none
    def drop_relation(self, relation):
        """Drop the given relation.

        *Implementors must call self.cache.drop() to preserve cache state!*

        :param self.Relation relation: The relation to drop
        """
        raise dbt.exceptions.NotImplementedException(
            '`drop_relation` is not implemented for this adapter!'
        )

    @abc.abstractmethod
    @available.parse_none
    def truncate_relation(self, relation):
        """Truncate the given relation.

        :param self.Relation relation: The relation to truncate
        """
        raise dbt.exceptions.NotImplementedException(
            '`truncate_relation` is not implemented for this adapter!'
        )

    @abc.abstractmethod
    @available.parse_none
    def rename_relation(self, from_relation, to_relation):
        """Rename the relation from from_relation to to_relation.

        Implementors must call self.cache.rename() to preserve cache state.

        :param self.Relation from_relation: The original relation name
        :param self.Relation to_relation: The new relation name
        """
        raise dbt.exceptions.NotImplementedException(
            '`rename_relation` is not implemented for this adapter!'
        )

    @abc.abstractmethod
    @available.parse_list
    def get_columns_in_relation(self, relation):
        """Get a list of the columns in the given Relation.

        :param self.Relation relation: The relation to query for.
        :return: Information about all columns in the given relation.
        :rtype: List[self.Column]
        """
        raise dbt.exceptions.NotImplementedException(
            '`get_columns_in_relation` is not implemented for this adapter!'
        )

    @available.deprecated('get_columns_in_relation', lambda *a, **k: [])
    def get_columns_in_table(self, schema, identifier):
        """DEPRECATED: Get a list of the columns in the given table."""
        relation = self.Relation.create(
            database=self.config.credentials.database,
            schema=schema,
            identifier=identifier,
            quote_policy=self.config.quoting
        )
        return self.get_columns_in_relation(relation)

    @abc.abstractmethod
    def expand_column_types(self, goal, current):
        """Expand the current table's types to match the goal table. (passable)

        :param self.Relation goal: A relation that currently exists in the
            database with columns of the desired types.
        :param self.Relation current: A relation that currently exists in the
            database with columns of unspecified types.
        """
        raise dbt.exceptions.NotImplementedException(
            '`expand_target_column_types` is not implemented for this adapter!'
        )

    @abc.abstractmethod
    def list_relations_without_caching(self, information_schema, schema):
        """List relations in the given schema, bypassing the cache.

        This is used as the underlying behavior to fill the cache.

        :param Relation information_schema: The information schema to list
            relations from.
        :param str schema: The name of the schema to list relations from.
        :return: The relations in schema
        :rtype: List[self.Relation]
        """
        raise dbt.exceptions.NotImplementedException(
            '`list_relations_without_caching` is not implemented for this '
            'adapter!'
        )

    ###
    # Provided methods about relations
    ###
    @available.parse_list
    def get_missing_columns(self, from_relation, to_relation):
        """Returns a list of Columns in from_relation that are missing from
        to_relation.

        :param Relation from_relation: The relation that might have extra
            columns
        :param Relation to_relation: The realtion that might have columns
            missing
        :return: The columns in from_relation that are missing from to_relation
        :rtype: List[self.Relation]
        """
        if not isinstance(from_relation, self.Relation):
            dbt.exceptions.invalid_type_error(
                method_name='get_missing_columns',
                arg_name='from_relation',
                got_value=from_relation,
                expected_type=self.Relation)

        if not isinstance(to_relation, self.Relation):
            dbt.exceptions.invalid_type_error(
                method_name='get_missing_columns',
                arg_name='to_relation',
                got_value=to_relation,
                expected_type=self.Relation)

        from_columns = {
            col.name: col for col in
            self.get_columns_in_relation(from_relation)
        }

        to_columns = {
            col.name: col for col in
            self.get_columns_in_relation(to_relation)
        }

        missing_columns = set(from_columns.keys()) - set(to_columns.keys())

        return [
            col for (col_name, col) in from_columns.items()
            if col_name in missing_columns
        ]

    @available.parse_none
    def valid_archive_target(self, relation):
        """Ensure that the target relation is valid, by making sure it has the
        expected columns.

        :param Relation relation: The relation to check
        :raises dbt.exceptions.CompilationException: If the columns are
            incorrect.
        """
        if not isinstance(relation, self.Relation):
            dbt.exceptions.invalid_type_error(
                method_name='is_existing_old_style_archive',
                arg_name='relation',
                got_value=relation,
                expected_type=self.Relation)

        columns = self.get_columns_in_relation(relation)
        names = set(c.name.lower() for c in columns)
        expanded_keys = ('scd_id', 'valid_from', 'valid_to')
        extra = []
        missing = []
        for legacy in expanded_keys:
            desired = 'dbt_' + legacy
            if desired not in names:
                missing.append(desired)
                if legacy in names:
                    extra.append(legacy)

        if missing:
            if extra:
                msg = (
                    'Archive target has ("{}") but not ("{}") - is it an '
                    'unmigrated previous version archive?'
                    .format('", "'.join(extra), '", "'.join(missing))
                )
            else:
                msg = (
                    'Archive target is not an archive table (missing "{}")'
                    .format('", "'.join(missing))
                )
            dbt.exceptions.raise_compiler_error(msg)

    @available.parse_none
    def expand_target_column_types(self, from_relation, to_relation):
        if not isinstance(from_relation, self.Relation):
            dbt.exceptions.invalid_type_error(
                method_name='expand_target_column_types',
                arg_name='from_relation',
                got_value=from_relation,
                expected_type=self.Relation)

        if not isinstance(to_relation, self.Relation):
            dbt.exceptions.invalid_type_error(
                method_name='expand_target_column_types',
                arg_name='to_relation',
                got_value=to_relation,
                expected_type=self.Relation)

        self.expand_column_types(from_relation, to_relation)

    def list_relations(self, database, schema):
        if self._schema_is_cached(database, schema):
            return self.cache.get_relations(database, schema)

        information_schema = self.Relation.create(
            database=database,
            schema=schema,
            model_name='',
            quote_policy=self.config.quoting
        ).information_schema()

        # we can't build the relations cache because we don't have a
        # manifest so we can't run any operations.
        relations = self.list_relations_without_caching(
            information_schema, schema
        )

        logger.debug('with database={}, schema={}, relations={}'
                     .format(database, schema, relations))
        return relations

    def _make_match_kwargs(self, database, schema, identifier):
        quoting = self.config.quoting
        if identifier is not None and quoting['identifier'] is False:
            identifier = identifier.lower()

        if schema is not None and quoting['schema'] is False:
            schema = schema.lower()

        if database is not None and quoting['database'] is False:
            database = database.lower()

        return filter_null_values({
            'database': database,
            'identifier': identifier,
            'schema': schema,
        })

    def _make_match(self, relations_list, database, schema, identifier):

        matches = []

        search = self._make_match_kwargs(database, schema, identifier)

        for relation in relations_list:
            if relation.matches(**search):
                matches.append(relation)

        return matches

    @available.parse_none
    def get_relation(self, database, schema, identifier):
        relations_list = self.list_relations(database, schema)

        matches = self._make_match(relations_list, database, schema,
                                   identifier)

        if len(matches) > 1:
            kwargs = {
                'identifier': identifier,
                'schema': schema,
                'database': database,
            }
            dbt.exceptions.get_relation_returned_multiple_results(
                kwargs, matches
            )

        elif matches:
            return matches[0]

        return None

    @available.deprecated('get_relation', lambda *a, **k: False)
    def already_exists(self, schema, name):
        """DEPRECATED: Return if a model already exists in the database"""
        database = self.config.credentials.database
        relation = self.get_relation(database, schema, name)
        return relation is not None

    ###
    # ODBC FUNCTIONS -- these should not need to change for every adapter,
    #                   although some adapters may override them
    ###
    @abc.abstractmethod
    @available.parse_none
    def create_schema(self, database, schema):
        """Create the given schema if it does not exist.

        :param str schema: The schema name to create.
        """
        raise dbt.exceptions.NotImplementedException(
            '`create_schema` is not implemented for this adapter!'
        )

    @abc.abstractmethod
    def drop_schema(self, database, schema):
        """Drop the given schema (and everything in it) if it exists.

        :param str schema: The schema name to drop.
        """
        raise dbt.exceptions.NotImplementedException(
            '`drop_schema` is not implemented for this adapter!'
        )

    @available
    @abstractclassmethod
    def quote(cls, identifier):
        """Quote the given identifier, as appropriate for the database.

        :param str identifier: The identifier to quote
        :return: The quoted identifier
        :rtype: str
        """
        raise dbt.exceptions.NotImplementedException(
            '`quote` is not implemented for this adapter!'
        )

    @available
    def quote_as_configured(self, identifier, quote_key):
        """Quote or do not quote the given identifer as configured in the
        project config for the quote key.

        The quote key should be one of 'database' (on bigquery, 'profile'),
        'identifier', or 'schema', or it will be treated as if you set `True`.
        """
        default = self.Relation.DEFAULTS['quote_policy'].get(quote_key)
        if self.config.quoting.get(quote_key, default):
            return self.quote(identifier)
        else:
            return identifier

    ###
    # Conversions: These must be implemented by concrete implementations, for
    # converting agate types into their sql equivalents.
    ###
    @abstractclassmethod
    def convert_text_type(cls, agate_table, col_idx):
        """Return the type in the database that best maps to the agate.Text
        type for the given agate table and column index.

        :param agate.Table agate_table: The table
        :param int col_idx: The index into the agate table for the column.
        :return: The name of the type in the database
        :rtype: str
        """
        raise dbt.exceptions.NotImplementedException(
            '`convert_text_type` is not implemented for this adapter!')

    @abstractclassmethod
    def convert_number_type(cls, agate_table, col_idx):
        """Return the type in the database that best maps to the agate.Number
        type for the given agate table and column index.

        :param agate.Table agate_table: The table
        :param int col_idx: The index into the agate table for the column.
        :return: The name of the type in the database
        :rtype: str
        """
        raise dbt.exceptions.NotImplementedException(
            '`convert_number_type` is not implemented for this adapter!')

    @abstractclassmethod
    def convert_boolean_type(cls, agate_table, col_idx):
        """Return the type in the database that best maps to the agate.Boolean
        type for the given agate table and column index.

        :param agate.Table agate_table: The table
        :param int col_idx: The index into the agate table for the column.
        :return: The name of the type in the database
        :rtype: str
        """
        raise dbt.exceptions.NotImplementedException(
            '`convert_boolean_type` is not implemented for this adapter!')

    @abstractclassmethod
    def convert_datetime_type(cls, agate_table, col_idx):
        """Return the type in the database that best maps to the agate.DateTime
        type for the given agate table and column index.

        :param agate.Table agate_table: The table
        :param int col_idx: The index into the agate table for the column.
        :return: The name of the type in the database
        :rtype: str
        """
        raise dbt.exceptions.NotImplementedException(
            '`convert_datetime_type` is not implemented for this adapter!')

    @abstractclassmethod
    def convert_date_type(cls, agate_table, col_idx):
        """Return the type in the database that best maps to the agate.Date
        type for the given agate table and column index.

        :param agate.Table agate_table: The table
        :param int col_idx: The index into the agate table for the column.
        :return: The name of the type in the database
        :rtype: str
        """
        raise dbt.exceptions.NotImplementedException(
            '`convert_date_type` is not implemented for this adapter!')

    @abstractclassmethod
    def convert_time_type(cls, agate_table, col_idx):
        """Return the type in the database that best maps to the
        agate.TimeDelta type for the given agate table and column index.

        :param agate.Table agate_table: The table
        :param int col_idx: The index into the agate table for the column.
        :return: The name of the type in the database
        :rtype: str
        """
        raise dbt.exceptions.NotImplementedException(
            '`convert_time_type` is not implemented for this adapter!')

    @available
    @classmethod
    def convert_type(cls, agate_table, col_idx):
        return cls.convert_agate_type(agate_table, col_idx)

    @classmethod
    def convert_agate_type(cls, agate_table, col_idx):
        agate_type = agate_table.column_types[col_idx]
        conversions = [
            (agate.Text, cls.convert_text_type),
            (agate.Number, cls.convert_number_type),
            (agate.Boolean, cls.convert_boolean_type),
            (agate.DateTime, cls.convert_datetime_type),
            (agate.Date, cls.convert_date_type),
            (agate.TimeDelta, cls.convert_time_type),
        ]
        for agate_cls, func in conversions:
            if isinstance(agate_type, agate_cls):
                return func(agate_table, col_idx)

    ###
    # Operations involving the manifest
    ###
    def execute_macro(self, macro_name, manifest=None, project=None,
                      context_override=None, kwargs=None, release=False):
        """Look macro_name up in the manifest and execute its results.

        :param str macro_name: The name of the macro to execute.
        :param Optional[Manifest] manifest: The manifest to use for generating
            the base macro execution context. If none is provided, use the
            internal manifest.
        :param Optional[str] project: The name of the project to search in, or
            None for the first match.
        :param Optional[dict] context_override: An optional dict to update()
            the macro execution context.
        :param Optional[dict] kwargs: An optional dict of keyword args used to
            pass to the macro.
        :param bool release: If True, release the connection after executing.

        Return an an AttrDict with three attributes: 'table', 'data', and
            'status'. 'table' is an agate.Table.
        """
        if kwargs is None:
            kwargs = {}
        if context_override is None:
            context_override = {}

        if manifest is None:
            manifest = self._internal_manifest

        macro = manifest.find_macro_by_name(macro_name, project)
        if macro is None:
            if project is None:
                package_name = 'any package'
            else:
                package_name = 'the "{}" package'.format(project)

            # The import of dbt.context.runtime below shadows 'dbt'
            import dbt.exceptions
            raise dbt.exceptions.RuntimeException(
                'dbt could not find a macro with the name "{}" in {}'
                .format(macro_name, package_name)
            )
        # This causes a reference cycle, as dbt.context.runtime.generate()
        # ends up calling get_adapter, so the import has to be here.
        import dbt.context.runtime
        macro_context = dbt.context.runtime.generate_macro(
            macro,
            self.config,
            manifest
        )
        macro_context.update(context_override)

        macro_function = macro.generator(macro_context)

        try:
            result = macro_function(**kwargs)
        finally:
            if release:
                self.release_connection()
        return result

    @classmethod
    def _catalog_filter_table(cls, table, manifest):
        """Filter the table as appropriate for catalog entries. Subclasses can
        override this to change filtering rules on a per-adapter basis.
        """
        return table.where(_catalog_filter_schemas(manifest))

    def get_catalog(self, manifest):
        """Get the catalog for this manifest by running the get catalog macro.
        Returns an agate.Table of catalog information.
        """
        information_schemas = list(self._get_cache_schemas(manifest).keys())
        # make it a list so macros can index into it.
        kwargs = {'information_schemas': information_schemas}
        table = self.execute_macro(GET_CATALOG_MACRO_NAME,
                                   kwargs=kwargs,
                                   release=True)

        results = self._catalog_filter_table(table, manifest)
        return results

    def cancel_open_connections(self):
        """Cancel all open connections."""
        return self.connections.cancel_open()

    def calculate_freshness(self, source, loaded_at_field, manifest=None):
        """Calculate the freshness of sources in dbt, and return it"""
        # in the future `source` will be a Relation instead of a string
        kwargs = {
            'source': source,
            'loaded_at_field': loaded_at_field
        }

        # run the macro
        table = self.execute_macro(
            FRESHNESS_MACRO_NAME,
            kwargs=kwargs,
            release=True,
            manifest=manifest
        )
        # now we have a 1-row table of the maximum `loaded_at_field` value and
        # the current time according to the db.
        if len(table) != 1 or len(table[0]) != 2:
            dbt.exceptions.raise_compiler_error(
                'Got an invalid result from "{}" macro: {}'.format(
                    FRESHNESS_MACRO_NAME, [tuple(r) for r in table]
                )
            )

        max_loaded_at = _utc(table[0][0], source, loaded_at_field)
        snapshotted_at = _utc(table[0][1], source, loaded_at_field)

        age = (snapshotted_at - max_loaded_at).total_seconds()
        return {
            'max_loaded_at': max_loaded_at,
            'snapshotted_at': snapshotted_at,
            'age': age,
        }
