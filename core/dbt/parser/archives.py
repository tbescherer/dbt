
from dbt.contracts.graph.parsed import ParsedArchiveNode
from dbt.node_types import NodeType
from dbt.parser.base_sql import BaseSqlParser, SQLParseResult
import dbt.clients.jinja
import dbt.exceptions
import dbt.utils


def set_archive_attributes(node):
    config_keys = {
        'target_database': 'database',
        'target_schema': 'schema'
    }

    for config_key, node_key in config_keys.items():
        if config_key in node.config:
            setattr(node, node_key, node.config[config_key])

    return node


class ArchiveBlockParser(BaseSqlParser):
    def parse_archives_from_file(self, file_node, tags=None):
        # the file node has a 'raw_sql' field that contains the jinja data with
        # (we hope!) `archive` blocks
        try:
            blocks = dbt.clients.jinja.extract_toplevel_blocks(
                file_node['raw_sql']
            )
        except dbt.exceptions.CompilationException as exc:
            if exc.node is None:
                exc.node = file_node
            raise
        for block in blocks:
            if block.block_type_name != NodeType.Archive:
                # non-archive blocks are just ignored
                continue
            name = block.block_name
            raw_sql = block.contents
            updates = {
                'raw_sql': raw_sql,
                'name': name,
            }
            yield dbt.utils.deep_merge(file_node, updates)

    @classmethod
    def get_compiled_path(cls, name, relative_path):
        return relative_path

    @classmethod
    def get_fqn(cls, node, package_project_config, extra=[]):
        parts = dbt.utils.split_path(node.path)
        fqn = [package_project_config.project_name]
        fqn.extend(parts[:-1])
        fqn.extend(extra)
        fqn.append(node.name)

        return fqn

    @staticmethod
    def validate_archives(node):
        if node.resource_type == NodeType.Archive:
            try:
                parsed_node = ParsedArchiveNode(**node.to_shallow_dict())
                return set_archive_attributes(parsed_node)

            except dbt.exceptions.JSONValidationException as exc:
                raise dbt.exceptions.CompilationException(str(exc), node)
        else:
            return node

    def parse_sql_nodes(self, nodes, tags=None):
        if tags is None:
            tags = []

        results = SQLParseResult()

        # in archives, we have stuff in blocks.
        for file_node in nodes:
            archive_nodes = list(
                self.parse_archives_from_file(file_node, tags=tags)
            )
            found = super(ArchiveBlockParser, self).parse_sql_nodes(
                nodes=archive_nodes, tags=tags
            )
            # make sure our blocks are going to work when we try to archive
            # them!
            found.parsed = {k: self.validate_archives(v) for
                            k, v in found.parsed.items()}

            results.update(found)
        return results
