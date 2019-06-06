from __future__ import print_function

import json

from dbt.task.runnable import GraphRunnableTask, ManifestTask
from dbt.node_types import NodeType
import dbt.exceptions
from dbt.logger import GLOBAL_LOGGER as logger
from dbt.logger import log_to_stderr


class ListTask(GraphRunnableTask):
    DEFAULT_RESOURCE_VALUES = frozenset((
        NodeType.Model,
        NodeType.Archive,
        NodeType.Seed,
        NodeType.Test,
        NodeType.Source,
    ))
    ALL_RESOURCE_VALUES = DEFAULT_RESOURCE_VALUES | frozenset((
        NodeType.Analysis,
    ))
    ALLOWED_KEYS = frozenset((
        'alias',
        'name',
        'package_name',
        'depends_on',
        'tags',
        'config',
        'resource_type',
        'source_name',
    ))

    def __init__(self, args, config):
        super(ListTask, self).__init__(args, config)
        self.args.single_threaded = True
        if self.args.models:
            if self.args.select:
                raise dbt.exceptions.RuntimeException(
                    '"models" and "select" are mutually exclusive arguments'
                )
            if self.args.resource_types:
                raise dbt.exceptions.RuntimeException(
                    '"models" and "resource_type" are mutually exclusive '
                    'arguments'
                )

    @classmethod
    def pre_init_hook(cls):
        """A hook called before the task is initialized."""
        log_to_stderr(logger)

    def _iterate_selected_nodes(self):
        nodes = sorted(self.select_nodes())
        if not nodes:
            logger.warning('No nodes selected!')
            return
        for node in nodes:
            yield self.manifest.nodes[node]

    def generate_selectors(self):
        for node in self._iterate_selected_nodes():
            selector = '.'.join(node.fqn)
            if node.resource_type == NodeType.Source:
                yield 'source:{}'.format(selector)
            else:
                yield selector

    def generate_names(self):
        for node in self._iterate_selected_nodes():
            if node.resource_type == NodeType.Source:
                yield '{0.source_name}.{0.name}'.format(node)
            else:
                yield node.name

    def generate_json(self):
        for node in self._iterate_selected_nodes():
            yield json.dumps({
                k: v
                for k, v in node.serialize().items()
                if k in self.ALLOWED_KEYS
            })

    def generate_paths(self):
        for node in self._iterate_selected_nodes():
            yield node.get('original_file_path')

    def run(self):
        ManifestTask._runtime_initialize(self)
        output = self.config.args.output
        if output == 'selector':
            generator = self.generate_selectors
        elif output == 'name':
            generator = self.generate_names
        elif output == 'json':
            generator = self.generate_json
        elif output == 'path':
            generator = self.generate_paths
        else:
            raise dbt.exceptions.IternalException(
                'Invalid output {}'.format(output)
            )
        for result in generator():
            self.node_results.append(result)
            print(result)
        return self.node_results

    @property
    def resource_types(self):
        if self.args.models:
            return [NodeType.Model]

        values = set(self.config.args.resource_types)
        if not values:
            return list(self.DEFAULT_RESOURCE_VALUES)

        if 'default' in values:
            values.remove('default')
            values.update(self.DEFAULT_RESOURCE_VALUES)
        if 'all' in values:
            values.remove('all')
            values.update(self.ALL_RESOURCE_VALUES)
        return list(values)

    @property
    def selector(self):
        if self.args.models:
            return self.args.models
        else:
            return self.args.select

    def build_query(self):
        return {
            "include": self.selector,
            "exclude": self.args.exclude,
            "resource_types": self.resource_types,
            "tags": [],
        }

    def interpret_results(self, results):
        return bool(results)
