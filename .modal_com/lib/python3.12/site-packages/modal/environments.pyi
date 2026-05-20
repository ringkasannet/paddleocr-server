import collections.abc
import google.protobuf.message
import modal._environments
import modal.client
import modal.object
import modal_proto.api_pb2
import synchronicity
import typing
import typing_extensions

class EnvironmentManager:
    """Namespace with methods for managing Environment objects."""
    def __init__(self, /, *args, **kwargs):
        """Initialize self.  See help(type(self)) for accurate signature."""
        ...

    class __create_spec(typing_extensions.Protocol):
        def __call__(
            self, /, name: str, *, restricted: bool = False, client: typing.Optional[modal.client.Client] = None
        ) -> None:
            """Create a new Environment.

            **Examples:**

            ```python notest
            modal.Environment.objects.create("my-environment")
            ```
            """
            ...

        async def aio(
            self, /, name: str, *, restricted: bool = False, client: typing.Optional[modal.client.Client] = None
        ) -> None:
            """Create a new Environment.

            **Examples:**

            ```python notest
            modal.Environment.objects.create("my-environment")
            ```
            """
            ...

    create: __create_spec

    class __list_spec(typing_extensions.Protocol):
        def __call__(self, /, *, client: typing.Optional[modal.client.Client] = None) -> list[Environment]:
            """Return a list of hydrated Environment objects.

            **Examples:**

            ```python notest
            environments = modal.Environment.objects.list()
            print([e.name for e in environments])
            ```
            """
            ...

        async def aio(self, /, *, client: typing.Optional[modal.client.Client] = None) -> list[Environment]:
            """Return a list of hydrated Environment objects.

            **Examples:**

            ```python notest
            environments = modal.Environment.objects.list()
            print([e.name for e in environments])
            ```
            """
            ...

    list: __list_spec

    class __delete_spec(typing_extensions.Protocol):
        def __call__(self, /, name: str, *, client: typing.Optional[modal.client.Client] = None) -> None:
            """Delete a named Environment.

            Warning: This is irreversible and will transitively delete all objects in the Environment.

            **Examples:**

            ```python notest
            modal.Environment.objects.delete("my-environment")
            ```
            """
            ...

        async def aio(self, /, name: str, *, client: typing.Optional[modal.client.Client] = None) -> None:
            """Delete a named Environment.

            Warning: This is irreversible and will transitively delete all objects in the Environment.

            **Examples:**

            ```python notest
            modal.Environment.objects.delete("my-environment")
            ```
            """
            ...

    delete: __delete_spec

class EnvironmentMembersManager:
    """mdmd:namespace
    Namespace with methods for managing the membership of a restricted Environment.

    See https://modal.com/docs/guide/rbac for more information on restricted Environments.
    """
    def __init__(self, environment: Environment):
        """mdmd:hidden"""
        ...

    class __list_spec(typing_extensions.Protocol):
        def __call__(
            self, /
        ) -> dict[typing.Literal["users", "service_users"], dict[str, typing.Literal["viewer", "contributor"]]]:
            """Return the members of a restricted Environment with their roles.

            **Examples:**

            ```python notest
            members = modal.Environment.from_name("my-restricted-env").members.list()
            print(members)
            # {
            #     "users": {"alice": "contributor", "bob": "viewer"},
            #     "service_users": {"alice-bot": "contributor"},
            # }
            ```
            """
            ...

        async def aio(
            self, /
        ) -> dict[typing.Literal["users", "service_users"], dict[str, typing.Literal["viewer", "contributor"]]]:
            """Return the members of a restricted Environment with their roles.

            **Examples:**

            ```python notest
            members = modal.Environment.from_name("my-restricted-env").members.list()
            print(members)
            # {
            #     "users": {"alice": "contributor", "bob": "viewer"},
            #     "service_users": {"alice-bot": "contributor"},
            # }
            ```
            """
            ...

    list: __list_spec

    class __update_spec(typing_extensions.Protocol):
        def __call__(
            self,
            /,
            *,
            users: typing.Optional[collections.abc.Mapping[str, typing.Literal["viewer", "contributor"]]] = None,
            service_users: typing.Optional[
                collections.abc.Mapping[str, typing.Literal["viewer", "contributor"]]
            ] = None,
        ) -> None:
            """Add or modify roles for members of a restricted Environment.

            Each user or service user will be added to the Environment if not currently a member;
            if already a member, the user or service user's role will be updated.

            **Examples:**

            ```python notest
            env = modal.Environment.from_name("my-restricted-env")
            env.members.update(
                users={"alice": "contributor", "bob": "viewer"},
                service_users={"alice-bot": "contributor"},
            )
            ```
            """
            ...

        async def aio(
            self,
            /,
            *,
            users: typing.Optional[collections.abc.Mapping[str, typing.Literal["viewer", "contributor"]]] = None,
            service_users: typing.Optional[
                collections.abc.Mapping[str, typing.Literal["viewer", "contributor"]]
            ] = None,
        ) -> None:
            """Add or modify roles for members of a restricted Environment.

            Each user or service user will be added to the Environment if not currently a member;
            if already a member, the user or service user's role will be updated.

            **Examples:**

            ```python notest
            env = modal.Environment.from_name("my-restricted-env")
            env.members.update(
                users={"alice": "contributor", "bob": "viewer"},
                service_users={"alice-bot": "contributor"},
            )
            ```
            """
            ...

    update: __update_spec

    class __remove_spec(typing_extensions.Protocol):
        def __call__(
            self,
            /,
            *,
            users: typing.Optional[collections.abc.Iterable[str]] = None,
            service_users: typing.Optional[collections.abc.Iterable[str]] = None,
        ) -> None:
            """Remove members from a restricted Environment.

            **Examples:**

            ```python notest
            env = modal.Environment.from_name("my-restricted-env")
            env.members.remove(
                users=["alice"],
                service_users=["alice-bot"],
            )
            ```
            """
            ...

        async def aio(
            self,
            /,
            *,
            users: typing.Optional[collections.abc.Iterable[str]] = None,
            service_users: typing.Optional[collections.abc.Iterable[str]] = None,
        ) -> None:
            """Remove members from a restricted Environment.

            **Examples:**

            ```python notest
            env = modal.Environment.from_name("my-restricted-env")
            env.members.remove(
                users=["alice"],
                service_users=["alice-bot"],
            )
            ```
            """
            ...

    remove: __remove_spec

    class ___dispatch_role_updates_spec(typing_extensions.Protocol):
        def __call__(self, /, requests: dict[str, modal_proto.api_pb2.EnvironmentRoleSetRequest]) -> None:
            """Send batched EnvironmentRoleSet RPCs and report all errors encountered."""
            ...

        async def aio(self, /, requests: dict[str, modal_proto.api_pb2.EnvironmentRoleSetRequest]) -> None:
            """Send batched EnvironmentRoleSet RPCs and report all errors encountered."""
            ...

    _dispatch_role_updates: ___dispatch_role_updates_spec

class Environment(modal.object.Object):
    _name: typing.Optional[str]
    _settings: modal._environments.EnvironmentSettings

    def __init__(self):
        """mdmd:hidden"""
        ...

    @property
    def name(self) -> typing.Optional[str]: ...
    @synchronicity.classproperty
    @classmethod
    def objects(cls) -> EnvironmentManager: ...
    @property
    def members(self) -> EnvironmentMembersManager: ...
    def _hydrate_metadata(self, metadata: google.protobuf.message.Message): ...
    @staticmethod
    def _get_or_create(
        name: str, repr: str, create_if_missing: bool = False, client: typing.Optional[modal.client.Client] = None
    ) -> Environment: ...
    @staticmethod
    def from_context(*, client: typing.Optional[modal.client.Client] = None) -> Environment:
        """Look up an Environment object using the current context.

        This method returns the Environment that is defined by the local configuration
        (i.e., your active profile or the `MODAL_ENVIRONMENT` environment variable), or
        it fetches the default environment from the server when not defined locally.
        If called inside a Modal container, it will return the Environment that container
        is associated with.
        """
        ...

    @staticmethod
    def from_name(
        name: str, *, create_if_missing: bool = False, client: typing.Optional[modal.client.Client] = None
    ) -> Environment:
        """Look up an Environment object using its name."""
        ...

class __create_environment_spec(typing_extensions.Protocol):
    def __call__(self, /, name: str, client: typing.Optional[modal.client.Client] = None): ...
    async def aio(self, /, name: str, client: typing.Optional[modal.client.Client] = None): ...

create_environment: __create_environment_spec

class __delete_environment_spec(typing_extensions.Protocol):
    def __call__(self, /, name: str, client: typing.Optional[modal.client.Client] = None): ...
    async def aio(self, /, name: str, client: typing.Optional[modal.client.Client] = None): ...

delete_environment: __delete_environment_spec

class __list_environments_spec(typing_extensions.Protocol):
    def __call__(
        self, /, client: typing.Optional[modal.client.Client] = None
    ) -> list[modal_proto.api_pb2.EnvironmentListItem]: ...
    async def aio(
        self, /, client: typing.Optional[modal.client.Client] = None
    ) -> list[modal_proto.api_pb2.EnvironmentListItem]: ...

list_environments: __list_environments_spec

class __update_environment_spec(typing_extensions.Protocol):
    def __call__(
        self,
        /,
        current_name: str,
        *,
        new_name: typing.Optional[str] = None,
        new_web_suffix: typing.Optional[str] = None,
        client: typing.Optional[modal.client.Client] = None,
    ): ...
    async def aio(
        self,
        /,
        current_name: str,
        *,
        new_name: typing.Optional[str] = None,
        new_web_suffix: typing.Optional[str] = None,
        client: typing.Optional[modal.client.Client] = None,
    ): ...

update_environment: __update_environment_spec
