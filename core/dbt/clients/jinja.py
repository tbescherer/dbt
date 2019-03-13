import codecs
import linecache
import os
import re

import jinja2
import jinja2._compat
import jinja2.compiler
import jinja2.ext
import jinja2.lexer
import jinja2.nodes
import jinja2.parser
import jinja2.sandbox

import dbt.compat
import dbt.exceptions

from dbt.node_types import NodeType
from dbt.utils import AttrDict

from dbt.logger import GLOBAL_LOGGER as logger  # noqa


class MacroFuzzParser(jinja2.parser.Parser):
    def parse_macro(self):
        node = jinja2.nodes.Macro(lineno=next(self.stream).lineno)

        # modified to fuzz macros defined in the same file. this way
        # dbt can understand the stack of macros being called.
        #  - @cmcarthur
        node.name = dbt.utils.get_dbt_macro_name(
            self.parse_assign_target(name_only=True).name)

        self.parse_signature(node)
        node.body = self.parse_statements(('name:endmacro',),
                                          drop_needle=True)
        return node

    def parse_statement(self):
        token = self.stream.current
        if token.type == 'data':
            # this is one of our custom data tokens, we should extract it
            ext = self.extensions.get(token.realtype)
            if ext is not None:
                return ext(self)
            self.fail_unknown_tag(token.realtype, token.lineno)
        return super(MacroFuzzParser, self).parse_statement()


class CustomCodeGenerator(jinja2.compiler.CodeGenerator):
    def visit_TemplateData(self, node, frame):
        if hasattr(node, 'realtype'):
            self.newline()
        super(CustomCodeGenerator, self).visit_TemplateData(node, frame)


def regex(x):
    return re.compile(x, re.M | re.S)


def _dbt_begin(name):
    return 'dbt_raw_{}_begin'.format(name)


def _dbt_end(name):
    return 'dbt_raw_{}_end'.format(name)


def _dbt_data(name):
    return 'dbt_raw_{}_data'.format(name)


def _dbt_ignore_tokens(names):
    for name in names:
        yield _dbt_begin(name)
        yield _dbt_end(name)


class BeginBlockToken(jinja2.lexer.Token):
    @property
    def type(self):
        return jinja2.lexer.TOKEN_BLOCK_BEGIN

    @property
    def realtype(self):
        trimleft = len('dbt_raw_')
        trimright = len('_begin')
        return self[1][trimleft:-trimright]


class EndBlockToken(jinja2.lexer.Token):
    @property
    def type(self):
        return jinja2.lexer.TOKEN_BLOCK_END

    @property
    def realtype(self):
        trimleft = len('dbt_raw_')
        trimright = len('_end')
        return self[1][trimleft:-trimright]


class BlockDataToken(jinja2.lexer.Token):
    @property
    def type(self):
        return jinja2.lexer.TOKEN_DATA

    @property
    def realtype(self):
        trimleft = len('dbt_raw_')
        trimright = len('_data')
        return self[1][trimleft:-trimright]


class JinjaCustomLexer(jinja2.lexer.Lexer):
    DBT_CUSTOM_BLOCK_NAMES = ['archive']
    DBT_IGNORE_TOKENS = tuple(_dbt_ignore_tokens(DBT_CUSTOM_BLOCK_NAMES))

    def __init__(self, environment):
        super(JinjaCustomLexer, self).__init__(environment)
        self.insert_custom_raw_rules(environment)

    def insert_custom_raw_rules(self, environment):
        # sooooo complicated to support this, skip it
        assert not environment.lstrip_blocks, 'lstrip_blocks not supported!'

        block_suffix_re = environment.trim_blocks and '\\n?' or ''
        block_prefix_re = '%s' % re.escape(environment.block_start_string)

        prefix_pattern = (
            r'(.*?)(?:\s*{escaped_start}\-|{block_prefix})\s*'
            r'{tag_name}\s*(?P<{token_name}>([A-Za-z_][A-Za-z_0-9]+))\s*'
            r'(?:\-{escaped_end}\s*|{escaped_end})'
        )

        suffix_pattern = (
            r'(.*?)((?:\s*{escaped_start}\-|{block_prefix})\s*end{tag_name}\s*'
            r'(?:\-{escaped_end}\s*|{escaped_end}{block_suffix}))'
        )

        for name in self.DBT_CUSTOM_BLOCK_NAMES:
            begin_token = _dbt_begin(name)
            end_token = _dbt_end(name)
            data_token = _dbt_data(name)

            this_prefix = prefix_pattern.format(
                token_name=begin_token,
                tag_name=name,
                escaped_start=re.escape(environment.block_start_string),
                escaped_end=re.escape(environment.block_end_string),
                block_prefix=block_prefix_re
            )
            root_rule = (
                regex(this_prefix),
                (data_token, '#bygroup'),
                '#bygroup'
            )

            self.rules['root'].insert(0, root_rule)

            this_suffix = suffix_pattern.format(
                tag_name=name,
                escaped_start=re.escape(environment.block_start_string),
                escaped_end=re.escape(environment.block_end_string),
                block_prefix=block_prefix_re,
                block_suffix=block_suffix_re
            )

            suffix = [
                (
                    regex(this_suffix),
                    (data_token, end_token),
                    '#pop'
                ),
                (
                    regex('(.)'),
                    (jinja2.lexer.Failure('Missing end of {} directive'
                                          .format(name)),),
                    None
                ),
            ]
            self.rules[begin_token] = suffix

    def wrap(self, stream, name=None, filename=None):

        superself = super(JinjaCustomLexer, self)
        begin_tokens = frozenset(
            _dbt_begin(n) for n in self.DBT_CUSTOM_BLOCK_NAMES
        )
        end_tokens = frozenset(
            _dbt_end(n) for n in self.DBT_CUSTOM_BLOCK_NAMES
        )
        data_tokens = frozenset(
            _dbt_data(n) for n in self.DBT_CUSTOM_BLOCK_NAMES
        )
        for token in superself.wrap(stream, name=name, filename=filename):
            if token.type in begin_tokens:
                yield BeginBlockToken(*token)
            elif token.type in end_tokens:
                yield EndBlockToken(*token)
            elif token.type in data_tokens:
                yield BlockDataToken(*token)
            else:
                yield token


class MacroFuzzEnvironment(jinja2.sandbox.SandboxedEnvironment):
    code_generator_class = CustomCodeGenerator

    def _parse(self, source, name, filename):
        return MacroFuzzParser(
            self, source, name,
            jinja2._compat.encode_filename(filename)
        ).parse()

    def _compile(self, source, filename):
        """Override jinja's compilation to stash the rendered source inside
        the python linecache for debugging.
        """
        if filename == '<template>':
            # make a better filename
            filename = 'dbt-{}'.format(
                codecs.encode(os.urandom(12), 'hex').decode('ascii')
            )
            # encode, though I don't think this matters
            filename = jinja2._compat.encode_filename(filename)
            # put ourselves in the cache
            linecache.cache[filename] = (
                len(source),
                None,
                [line+'\n' for line in source.splitlines()],
                filename
            )

        return super(MacroFuzzEnvironment, self)._compile(source, filename)

    # TODO: this should actually be cached like jinja does
    lexer = property(JinjaCustomLexer, 'The dbt custom lexer')


class TemplateCache(object):
    def __init__(self):
        self.file_cache = {}

    def get_node_template(self, node):
        key = (node['package_name'], node['original_file_path'])

        if key in self.file_cache:
            return self.file_cache[key]

        template = get_template(
            string=node.get('raw_sql'),
            ctx={},
            node=node
        )
        self.file_cache[key] = template

        return template

    def clear(self):
        self.file_cache.clear()


template_cache = TemplateCache()


def macro_generator(node):
    def apply_context(context):
        def call(*args, **kwargs):
            name = node.get('name')
            template = template_cache.get_node_template(node)
            module = template.make_module(context, False, context)

            macro = module.__dict__[dbt.utils.get_dbt_macro_name(name)]
            module.__dict__.update(context)

            try:
                return macro(*args, **kwargs)
            except dbt.exceptions.MacroReturn as e:
                return e.value
            except (TypeError, jinja2.exceptions.TemplateRuntimeError) as e:
                dbt.exceptions.raise_compiler_error(str(e), node)
            except dbt.exceptions.CompilationException as e:
                e.stack.append(node)
                raise e

        return call
    return apply_context


class MaterializationExtension(jinja2.ext.Extension):
    tags = ['materialization']

    def parse(self, parser):
        node = jinja2.nodes.Macro(lineno=next(parser.stream).lineno)
        materialization_name = \
            parser.parse_assign_target(name_only=True).name

        adapter_name = 'default'
        node.args = []
        node.defaults = []

        while parser.stream.skip_if('comma'):
            target = parser.parse_assign_target(name_only=True)

            if target.name == 'default':
                pass

            elif target.name == 'adapter':
                parser.stream.expect('assign')
                value = parser.parse_expression()
                adapter_name = value.value

            else:
                dbt.exceptions.invalid_materialization_argument(
                    materialization_name, target.name)

        node.name = dbt.utils.get_materialization_macro_name(
            materialization_name, adapter_name)

        node.body = parser.parse_statements(('name:endmaterialization',),
                                            drop_needle=True)

        return node


class DocumentationExtension(jinja2.ext.Extension):
    tags = ['docs']

    def parse(self, parser):
        node = jinja2.nodes.Macro(lineno=next(parser.stream).lineno)
        docs_name = parser.parse_assign_target(name_only=True).name

        node.args = []
        node.defaults = []
        node.name = dbt.utils.get_docs_macro_name(docs_name)
        node.body = parser.parse_statements(('name:enddocs',),
                                            drop_needle=True)
        return node


class ArchiveExtension(jinja2.ext.Extension):
    tags = ['archive']

    def _is_my_begin_tag(self, token):
        return (isinstance(token, BeginBlockToken) and
                token.realtype == self.tags[0])

    def _is_my_end_tag(self, token):
        return (isinstance(token, EndBlockToken) and
                token.realtype == self.tags[0])

    def filter_stream(self, stream):
        for token in stream:
            if self._is_my_begin_tag(token):
                # yeield a begin tag
                yield token
                # now yield a name field for the begin tag. In our case, that
                # will be 'archive', so we can get called by parse()
                yield jinja2.lexer.Token(token.lineno, 'name', self.tags[0])
                # our parse() method will see this and the data field
                yield jinja2.lexer.Token(token.lineno, 'name', token.value)
            elif self._is_my_end_tag(token):
                yield token
            else:
                yield token

    def parse(self, parser):
        node = jinja2.nodes.Macro(lineno=next(parser.stream).lineno)
        archive_name = parser.parse_assign_target(name_only=True).name
        node.args = []
        node.defaults = []
        node.name = dbt.utils.get_archive_macro_name(archive_name)
        token = parser.stream.expect('data')
        data = jinja2.nodes.TemplateData(token.value,
                                         lineno=token.lineno)
        data.realtype = self.tags[0]
        node.body = [data]
        return node


def _is_dunder_name(name):
    return name.startswith('__') and name.endswith('__')


def create_macro_capture_env(node):

    class ParserMacroCapture(jinja2.Undefined):
        """
        This class sets up the parser to capture macros.
        """
        def __init__(self, hint=None, obj=None, name=None, exc=None):
            super(ParserMacroCapture, self).__init__(hint=hint, name=name)
            self.node = node
            self.name = name
            self.package_name = node.get('package_name')
            # jinja uses these for safety, so we have to override them.
            # see https://github.com/pallets/jinja/blob/master/jinja2/sandbox.py#L332-L339 # noqa
            self.unsafe_callable = False
            self.alters_data = False

        def __deepcopy__(self, memo):
            path = os.path.join(self.node.get('root_path'),
                                self.node.get('original_file_path'))

            logger.debug(
                'dbt encountered an undefined variable, "{}" in node {}.{} '
                '(source path: {})'
                .format(self.name, self.node.get('package_name'),
                        self.node.get('name'), path))

            # match jinja's message
            dbt.exceptions.raise_compiler_error(
                "{!r} is undefined".format(self.name),
                node=self.node
            )

        def __getitem__(self, name):
            # Propagate the undefined value if a caller accesses this as if it
            # were a dictionary
            return self

        def __getattr__(self, name):
            if name == 'name' or _is_dunder_name(name):
                raise AttributeError(
                    "'{}' object has no attribute '{}'"
                    .format(type(self).__name__, name)
                )

            self.package_name = self.name
            self.name = name

            return self

        def __call__(self, *args, **kwargs):
            return True

    return ParserMacroCapture


def get_environment(node=None, capture_macros=False):
    args = {
        'extensions': ['jinja2.ext.do']
    }

    if capture_macros:
        args['undefined'] = create_macro_capture_env(node)

    args['extensions'].append(MaterializationExtension)
    args['extensions'].append(DocumentationExtension)
    args['extensions'].append(ArchiveExtension)

    return MacroFuzzEnvironment(**args)


def parse(string):
    try:
        return get_environment().parse(dbt.compat.to_string(string))

    except (jinja2.exceptions.TemplateSyntaxError,
            jinja2.exceptions.UndefinedError) as e:
        e.translated = False
        dbt.exceptions.raise_compiler_error(str(e))


def get_template(string, ctx, node=None, capture_macros=False):
    try:
        env = get_environment(node, capture_macros)

        template_source = dbt.compat.to_string(string)
        return env.from_string(template_source, globals=ctx)

    except (jinja2.exceptions.TemplateSyntaxError,
            jinja2.exceptions.UndefinedError) as e:
        e.translated = False
        dbt.exceptions.raise_compiler_error(str(e), node)


def render_template(template, ctx, node=None):
    try:
        return template.render(ctx)

    except (jinja2.exceptions.TemplateSyntaxError,
            jinja2.exceptions.UndefinedError) as e:
        e.translated = False
        dbt.exceptions.raise_compiler_error(str(e), node)


def get_rendered(string, ctx, node=None,
                 capture_macros=False):
    template = get_template(string, ctx, node,
                            capture_macros=capture_macros)

    return render_template(template, ctx, node)


def undefined_error(msg):
    raise jinja2.exceptions.UndefinedError(msg)
