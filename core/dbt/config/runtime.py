
from copy import deepcopy
import pprint

from dbt.utils import parse_cli_vars
from dbt.contracts.project import Configuration
from dbt.exceptions import DbtProjectError
from dbt.exceptions import ValidationException
from dbt.adapters.factory import get_relation_class_by_name

from .profile import Profile
from .project import Project


_ARCHIVE_REMOVED_MESSAGE = '''
The `archive` section in `dbt_project.yml` is no longer supported. Please use a
`snapshot` block instead. For more information on snapshot blocks and a script
to help migrate these archives, please consult the 0.14.0 migration guide:

https://docs.getdbt.com/v0.14/docs/upgrading-to-014
'''.strip()


class RuntimeConfig(Project, Profile):
    """The runtime configuration, as constructed from its components. There's a
    lot because there is a lot of stuff!
    """
    def __init__(self, project_name, version, project_root, source_paths,
                 macro_paths, data_paths, test_paths, analysis_paths,
                 docs_paths, target_path, archive_paths, clean_targets,
                 log_path, modules_path, quoting, models, on_run_start,
                 on_run_end, archive, seeds, dbt_version, profile_name,
                 target_name, config, threads, credentials, packages, args):
        # 'vars'
        self.args = args
        self.cli_vars = parse_cli_vars(getattr(args, 'vars', '{}'))
        # 'project'
        Project.__init__(
            self,
            project_name=project_name,
            version=version,
            project_root=project_root,
            profile_name=profile_name,
            source_paths=source_paths,
            macro_paths=macro_paths,
            data_paths=data_paths,
            test_paths=test_paths,
            analysis_paths=analysis_paths,
            docs_paths=docs_paths,
            target_path=target_path,
            archive_paths=archive_paths,
            clean_targets=clean_targets,
            log_path=log_path,
            modules_path=modules_path,
            quoting=quoting,
            models=models,
            on_run_start=on_run_start,
            on_run_end=on_run_end,
            archive=archive,
            seeds=seeds,
            dbt_version=dbt_version,
            packages=packages
        )
        # 'profile'
        Profile.__init__(
            self,
            profile_name=profile_name,
            target_name=target_name,
            config=config,
            threads=threads,
            credentials=credentials
        )
        self.validate()

    @classmethod
    def from_parts(cls, project, profile, args, allow_archive_configs=False):
        """Instantiate a RuntimeConfig from its components.

        :param profile Profile: A parsed dbt Profile.
        :param project Project: A parsed dbt Project.
        :param args argparse.Namespace: The parsed command-line arguments.
        :param allow_archive_configs bool: If True, ignore archive blocks in
            configs. This flag exists to enable archive migration.
        :returns RuntimeConfig: The new configuration.
        """
        quoting = deepcopy(
            get_relation_class_by_name(profile.credentials.type)
            .DEFAULTS['quote_policy']
        )
        quoting.update(project.quoting)
        if project.archive and not allow_archive_configs:
            # if the user has an `archive` section, raise an error
            raise DbtProjectError(_ARCHIVE_REMOVED_MESSAGE)

        return cls(
            project_name=project.project_name,
            version=project.version,
            project_root=project.project_root,
            source_paths=project.source_paths,
            macro_paths=project.macro_paths,
            data_paths=project.data_paths,
            test_paths=project.test_paths,
            analysis_paths=project.analysis_paths,
            docs_paths=project.docs_paths,
            target_path=project.target_path,
            archive_paths=project.archive_paths,
            clean_targets=project.clean_targets,
            log_path=project.log_path,
            modules_path=project.modules_path,
            quoting=quoting,
            models=project.models,
            on_run_start=project.on_run_start,
            on_run_end=project.on_run_end,
            archive=project.archive,
            seeds=project.seeds,
            dbt_version=project.dbt_version,
            packages=project.packages,
            profile_name=profile.profile_name,
            target_name=profile.target_name,
            config=profile.config,
            threads=profile.threads,
            credentials=profile.credentials,
            args=args
        )

    def new_project(self, project_root):
        """Given a new project root, read in its project dictionary, supply the
        existing project's profile info, and create a new project file.

        :param project_root str: A filepath to a dbt project.
        :raises DbtProfileError: If the profile is invalid.
        :raises DbtProjectError: If project is missing or invalid.
        :returns RuntimeConfig: The new configuration.
        """
        # copy profile
        profile = Profile(**self.to_profile_info())
        profile.validate()
        # load the new project and its packages. Don't pass cli variables.
        project = Project.from_project_root(project_root, {})

        cfg = self.from_parts(
            project=project,
            profile=profile,
            args=deepcopy(self.args),
        )
        # force our quoting back onto the new project.
        cfg.quoting = deepcopy(self.quoting)
        return cfg

    def serialize(self):
        """Serialize the full configuration to a single dictionary. For any
        instance that has passed validate() (which happens in __init__), it
        matches the Configuration contract.

        Note that args are not serialized.

        :returns dict: The serialized configuration.
        """
        result = self.to_project_config(with_packages=True)
        result.update(self.to_profile_info(serialize_credentials=True))
        result['cli_vars'] = deepcopy(self.cli_vars)
        return result

    def __str__(self):
        return pprint.pformat(self.serialize())

    def validate(self):
        """Validate the configuration against its contract.

        :raises DbtProjectError: If the configuration fails validation.
        """
        try:
            Configuration(**self.serialize())
        except ValidationException as e:
            raise DbtProjectError(str(e))

        if getattr(self.args, 'version_check', False):
            self.validate_version()

    @classmethod
    def from_args(cls, args, allow_archive_configs=False):
        """Given arguments, read in dbt_project.yml from the current directory,
        read in packages.yml if it exists, and use them to find the profile to
        load.

        :param args argparse.Namespace: The arguments as parsed from the cli.
        :param allow_archive_configs bool: If True, ignore archive blocks in
            configs. This flag exists to enable archive migration.
        :raises DbtProjectError: If the project is invalid or missing.
        :raises DbtProfileError: If the profile is invalid or missing.
        :raises ValidationException: If the cli variables are invalid.
        """
        # build the project and read in packages.yml
        project = Project.from_args(args)

        # build the profile
        profile = Profile.from_args(
            args=args,
            project_profile_name=project.profile_name
        )

        return cls.from_parts(
            project=project,
            profile=profile,
            args=args,
            allow_archive_configs=allow_archive_configs
        )
