import enum
import os
import typing
import typing_extensions

class FileType(enum.Enum):
    """Type of a filesystem entry."""

    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"

class FileInfo:
    """Metadata for a file or directory entry in a Sandbox."""

    name: str
    path: str
    type: FileType
    size: int
    mode: int
    permissions: str
    owner: str
    group: str
    modified_time: float
    symlink_target: typing.Optional[str]

    def is_file(self) -> bool:
        """Return `True` if this entry is a regular file."""
        ...

    def is_dir(self) -> bool:
        """Return `True` if this entry is a directory."""
        ...

    def is_symlink(self) -> bool:
        """Return `True` if this entry is a symbolic link."""
        ...

    def __init__(
        self,
        name: str,
        path: str,
        type: FileType,
        size: int,
        mode: int,
        permissions: str,
        owner: str,
        group: str,
        modified_time: float,
        symlink_target: typing.Optional[str],
    ) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
        ...

    def __repr__(self):
        """Return repr(self)."""
        ...

    def __eq__(self, other):
        """Return self==value."""
        ...

    def __setattr__(self, name, value):
        """Implement setattr(self, name, value)."""
        ...

    def __delattr__(self, name):
        """Implement delattr(self, name)."""
        ...

    def __hash__(self):
        """Return hash(self)."""
        ...

def _log_throughput(op: str, size_bytes: int, dur_s: float) -> None: ...

class _SandboxFilesystem:
    """mdmd:namespace
    Namespace for Sandbox filesystem APIs.
    """
    def __init__(self, sandbox):
        """mdmd:hidden"""
        ...

    async def copy_from_local(self, local_path: typing.Union[str, os.PathLike], remote_path: str) -> None:
        """Copy a local file into the Sandbox.

        `remote_path` must be an absolute path to a file in the Sandbox.
        Parent directories for `remote_path` are created if needed.
        The remote file is overwritten if it already exists.

        **Raises**

        - `SandboxFilesystemNotADirectoryError`: a parent path component of `remote_path` is not a directory.
        - `SandboxFilesystemIsADirectoryError`: `remote_path` points to a directory.
        - `SandboxFilesystemPermissionError`: write permission is denied in the Sandbox.
        - `SandboxFilesystemError`: the command fails for any other reason.
        - `FileNotFoundError`: `local_path` does not exist.
        - `IsADirectoryError`: `local_path` is a directory.
        - `PermissionError`: reading `local_path` is not permitted.

        **Usage**

        ```python fixture:sandbox fixture:tmpdir
        import tempfile
        from pathlib import Path

        local_path = Path(tempfile.mktemp())
        local_path.write_text("Hello, world!\n")
        sandbox.filesystem.copy_from_local(local_path, "/tmp/hello.txt")
        ```
        """
        ...

    async def copy_to_local(self, remote_path: str, local_path: typing.Union[str, os.PathLike]) -> None:
        """Copy a file from the Sandbox to a local path.

        `remote_path` must be an absolute path to a file in the Sandbox.
        Parent directories for `local_path` are created if needed.
        The local file is overwritten if it already exists.

        **Raises**

        - `SandboxFilesystemNotFoundError`: the remote path does not exist.
        - `SandboxFilesystemIsADirectoryError`: the remote path points to a directory.
        - `SandboxFilesystemPermissionError`: read permission is denied in the Sandbox.
        - `SandboxFilesystemError`: the command fails for any other reason.
        - `IsADirectoryError`: `local_path` points to a directory.
        - `NotADirectoryError`: a component of the `local_path` parent is not a directory.
        - `PermissionError`: writing `local_path` is not permitted.

        **Usage**

        ```python fixture:sandbox fixture:tmpdir
        sandbox.filesystem.write_text("Hello, world!\n", "/tmp/hello.txt")
        sandbox.filesystem.copy_to_local("/tmp/hello.txt", "/tmp/local-hello.txt")
        ```
        """
        ...

    async def list_files(self, remote_path: str) -> list[FileInfo]:
        """List files and directories in a Sandbox directory.

        `remote_path` must be an absolute path to a directory in the Sandbox.
        Returns a list of `FileInfo` objects describing each entry.

        **Raises**

        - `SandboxFilesystemNotFoundError`: the path does not exist.
        - `SandboxFilesystemNotADirectoryError`: the path is not a directory.
        - `SandboxFilesystemPermissionError`: read permission is denied.
        - `SandboxFilesystemError`: the command fails for any other reason.

        **Usage**

        ```python fixture:sandbox
        entries = sandbox.filesystem.list_files("/tmp")
        for entry in entries:
            print(entry.name, entry.type, entry.size)
        ```
        """
        ...

    async def make_directory(self, remote_path: str, *, create_parents: bool = True) -> None:
        """Create a new directory in the Sandbox.

        `remote_path` must be an absolute path in the Sandbox.

        When `create_parents` is `True` (the default), any missing parent directories are created and the call is
        idempotent (succeeds silently if the directory already exists). When `create_parents` is `False`, the
        immediate parent directory must already exist and the path must not already exist.

        **Raises**

        - `SandboxFilesystemNotFoundError`: the parent directory does not exist and `create_parents` is `False`.
        - `SandboxFilesystemPathAlreadyExistsError`: the path already exists.
        - `SandboxFilesystemNotADirectoryError`: a path component is not a directory.
        - `SandboxFilesystemPermissionError`: creation is not permitted.
        - `InvalidError`: the operation is not supported by the mount.
        - `SandboxFilesystemError`: the command fails for any other reason.

        **Usage**

        ```python fixture:sandbox
        sandbox.filesystem.make_directory("/tmp/a/b/c")
        ```
        """
        ...

    async def read_bytes(self, remote_path: str) -> bytes:
        """Read a file from the Sandbox and return its contents as bytes.

        `remote_path` must be an absolute path to a file in the Sandbox.

        **Raises**

        - `SandboxFilesystemNotFoundError`: the path does not exist.
        - `SandboxFilesystemIsADirectoryError`: the path points to a directory.
        - `SandboxFilesystemPermissionError`: read permission is denied.
        - `SandboxFilesystemError`: the command fails for any other reason.

        **Usage**

        ```python fixture:sandbox
        sandbox.filesystem.write_bytes(b"Hello, world!\n", "/tmp/hello.bin")
        contents = sandbox.filesystem.read_bytes("/tmp/hello.bin")
        print(contents.decode("utf-8"))
        ```
        """
        ...

    async def read_text(self, remote_path: str) -> str:
        """Read a file from the Sandbox and return its contents as a UTF-8 string.

        `remote_path` must be an absolute path to a file in the Sandbox.

        **Raises**

        - `SandboxFilesystemNotFoundError`: the path does not exist.
        - `SandboxFilesystemIsADirectoryError`: the path points to a directory.
        - `SandboxFilesystemPermissionError`: read permission is denied.
        - `SandboxFilesystemError`: the command fails for any other reason.

        **Usage**

        ```python fixture:sandbox
        sandbox.filesystem.write_text("Hello, world!\n", "/tmp/hello.txt")
        contents = sandbox.filesystem.read_text("/tmp/hello.txt")
        print(contents)
        ```
        """
        ...

    async def remove(self, remote_path: str, *, recursive: bool = False) -> None:
        """Remove a file or directory in the Sandbox.

        `remote_path` must be an absolute path in the Sandbox.

        When `remote_path` is a directory and `recursive` is `False` (the
        default), removes it only if it is empty. When `recursive` is `True`,
        removes the directory and all its contents.

        Recursive directory removal is not supported on all mounts.
        In particular, `CloudBucketMount` does not support it. An
        `InvalidError` is raised in that case.

        **Raises**

        - `SandboxFilesystemNotFoundError`: the path does not exist.
        - `SandboxFilesystemDirectoryNotEmptyError`: `recursive` is `False` and the directory is not empty.
        - `SandboxFilesystemPermissionError`: removal is not permitted.
        - `InvalidError`: the operation is not supported by the mount.
        - `SandboxFilesystemError`: the command fails for any other reason.

        **Usage**

        To remove a file:

        ```python fixture:sandbox
        sandbox.filesystem.write_bytes(b"Hello, world!\n", "/tmp/hello.bin")
        sandbox.filesystem.remove("/tmp/hello.bin")
        ```

        To remove a directory and all its contents:

        ```python fixture:sandbox
        sandbox.filesystem.make_directory("/tmp/mydir/subdir")
        sandbox.filesystem.remove("/tmp/mydir", recursive=True)
        ```
        """
        ...

    async def stat(self, remote_path: str) -> FileInfo:
        """Return metadata for a single file, directory, or symlink in the Sandbox.

        `remote_path` must be an absolute path in the Sandbox. If `remote_path` is a symlink, the returned
        `FileInfo` object describes the symlink, not the target it points to.

        **Raises**

        - `SandboxFilesystemNotFoundError`: the path does not exist.
        - `SandboxFilesystemNotADirectoryError`: a non-leaf component of the path is not a directory.
        - `SandboxFilesystemPermissionError`: a component of the path is not searchable.
        - `SandboxFilesystemError`: the command fails for any other reason.

        **Usage**

        ```python fixture:sandbox
        sandbox.filesystem.write_text("Hello, world!\n", "/tmp/hello.txt")
        info = sandbox.filesystem.stat("/tmp/hello.txt")
        print(info.size, info.permissions, info.modified_time)
        ```
        """
        ...

    async def write_bytes(self, data: typing.Union[bytes, bytearray, memoryview], remote_path: str) -> None:
        """Write binary content to a file in the Sandbox.

        `remote_path` must be an absolute path to a file in the Sandbox.
        Parent directories for `remote_path` are created if needed.
        The remote file is overwritten if it already exists.

        **Raises**

        - `SandboxFilesystemNotADirectoryError`: a parent path component is not a directory.
        - `SandboxFilesystemIsADirectoryError`: `remote_path` points to a directory.
        - `SandboxFilesystemPermissionError`: write permission is denied.
        - `SandboxFilesystemError`: the command fails for any other reason.

        **Usage**

        ```python fixture:sandbox
        sandbox.filesystem.write_bytes(b"Hello, world!\n", "/tmp/hello.bin")
        ```
        """
        ...

    async def write_text(self, data: str, remote_path: str) -> None:
        """Write UTF-8 text to a file in the Sandbox.

        `remote_path` must be an absolute path to a file in the Sandbox.
        Parent directories for `remote_path` are created if needed.
        The remote file is overwritten if it already exists.

        **Raises**

        - `SandboxFilesystemNotADirectoryError`: a parent path component is not a directory.
        - `SandboxFilesystemIsADirectoryError`: `remote_path` points to a directory.
        - `SandboxFilesystemPermissionError`: write permission is denied.
        - `SandboxFilesystemError`: the command fails for any other reason.

        **Usage**

        ```python fixture:sandbox
        sandbox.filesystem.write_text("Hello, world!\n", "/tmp/hello.txt")
        ```
        """
        ...

class SandboxFilesystem:
    """mdmd:namespace
    Namespace for Sandbox filesystem APIs.
    """
    def __init__(self, sandbox):
        """mdmd:hidden"""
        ...

    class __copy_from_local_spec(typing_extensions.Protocol):
        def __call__(self, /, local_path: typing.Union[str, os.PathLike], remote_path: str) -> None:
            """Copy a local file into the Sandbox.

            `remote_path` must be an absolute path to a file in the Sandbox.
            Parent directories for `remote_path` are created if needed.
            The remote file is overwritten if it already exists.

            **Raises**

            - `SandboxFilesystemNotADirectoryError`: a parent path component of `remote_path` is not a directory.
            - `SandboxFilesystemIsADirectoryError`: `remote_path` points to a directory.
            - `SandboxFilesystemPermissionError`: write permission is denied in the Sandbox.
            - `SandboxFilesystemError`: the command fails for any other reason.
            - `FileNotFoundError`: `local_path` does not exist.
            - `IsADirectoryError`: `local_path` is a directory.
            - `PermissionError`: reading `local_path` is not permitted.

            **Usage**

            ```python fixture:sandbox fixture:tmpdir
            import tempfile
            from pathlib import Path

            local_path = Path(tempfile.mktemp())
            local_path.write_text("Hello, world!\n")
            sandbox.filesystem.copy_from_local(local_path, "/tmp/hello.txt")
            ```
            """
            ...

        async def aio(self, /, local_path: typing.Union[str, os.PathLike], remote_path: str) -> None:
            """Copy a local file into the Sandbox.

            `remote_path` must be an absolute path to a file in the Sandbox.
            Parent directories for `remote_path` are created if needed.
            The remote file is overwritten if it already exists.

            **Raises**

            - `SandboxFilesystemNotADirectoryError`: a parent path component of `remote_path` is not a directory.
            - `SandboxFilesystemIsADirectoryError`: `remote_path` points to a directory.
            - `SandboxFilesystemPermissionError`: write permission is denied in the Sandbox.
            - `SandboxFilesystemError`: the command fails for any other reason.
            - `FileNotFoundError`: `local_path` does not exist.
            - `IsADirectoryError`: `local_path` is a directory.
            - `PermissionError`: reading `local_path` is not permitted.

            **Usage**

            ```python fixture:sandbox fixture:tmpdir
            import tempfile
            from pathlib import Path

            local_path = Path(tempfile.mktemp())
            local_path.write_text("Hello, world!\n")
            sandbox.filesystem.copy_from_local(local_path, "/tmp/hello.txt")
            ```
            """
            ...

    copy_from_local: __copy_from_local_spec

    class __copy_to_local_spec(typing_extensions.Protocol):
        def __call__(self, /, remote_path: str, local_path: typing.Union[str, os.PathLike]) -> None:
            """Copy a file from the Sandbox to a local path.

            `remote_path` must be an absolute path to a file in the Sandbox.
            Parent directories for `local_path` are created if needed.
            The local file is overwritten if it already exists.

            **Raises**

            - `SandboxFilesystemNotFoundError`: the remote path does not exist.
            - `SandboxFilesystemIsADirectoryError`: the remote path points to a directory.
            - `SandboxFilesystemPermissionError`: read permission is denied in the Sandbox.
            - `SandboxFilesystemError`: the command fails for any other reason.
            - `IsADirectoryError`: `local_path` points to a directory.
            - `NotADirectoryError`: a component of the `local_path` parent is not a directory.
            - `PermissionError`: writing `local_path` is not permitted.

            **Usage**

            ```python fixture:sandbox fixture:tmpdir
            sandbox.filesystem.write_text("Hello, world!\n", "/tmp/hello.txt")
            sandbox.filesystem.copy_to_local("/tmp/hello.txt", "/tmp/local-hello.txt")
            ```
            """
            ...

        async def aio(self, /, remote_path: str, local_path: typing.Union[str, os.PathLike]) -> None:
            """Copy a file from the Sandbox to a local path.

            `remote_path` must be an absolute path to a file in the Sandbox.
            Parent directories for `local_path` are created if needed.
            The local file is overwritten if it already exists.

            **Raises**

            - `SandboxFilesystemNotFoundError`: the remote path does not exist.
            - `SandboxFilesystemIsADirectoryError`: the remote path points to a directory.
            - `SandboxFilesystemPermissionError`: read permission is denied in the Sandbox.
            - `SandboxFilesystemError`: the command fails for any other reason.
            - `IsADirectoryError`: `local_path` points to a directory.
            - `NotADirectoryError`: a component of the `local_path` parent is not a directory.
            - `PermissionError`: writing `local_path` is not permitted.

            **Usage**

            ```python fixture:sandbox fixture:tmpdir
            sandbox.filesystem.write_text("Hello, world!\n", "/tmp/hello.txt")
            sandbox.filesystem.copy_to_local("/tmp/hello.txt", "/tmp/local-hello.txt")
            ```
            """
            ...

    copy_to_local: __copy_to_local_spec

    class __list_files_spec(typing_extensions.Protocol):
        def __call__(self, /, remote_path: str) -> list[FileInfo]:
            """List files and directories in a Sandbox directory.

            `remote_path` must be an absolute path to a directory in the Sandbox.
            Returns a list of `FileInfo` objects describing each entry.

            **Raises**

            - `SandboxFilesystemNotFoundError`: the path does not exist.
            - `SandboxFilesystemNotADirectoryError`: the path is not a directory.
            - `SandboxFilesystemPermissionError`: read permission is denied.
            - `SandboxFilesystemError`: the command fails for any other reason.

            **Usage**

            ```python fixture:sandbox
            entries = sandbox.filesystem.list_files("/tmp")
            for entry in entries:
                print(entry.name, entry.type, entry.size)
            ```
            """
            ...

        async def aio(self, /, remote_path: str) -> list[FileInfo]:
            """List files and directories in a Sandbox directory.

            `remote_path` must be an absolute path to a directory in the Sandbox.
            Returns a list of `FileInfo` objects describing each entry.

            **Raises**

            - `SandboxFilesystemNotFoundError`: the path does not exist.
            - `SandboxFilesystemNotADirectoryError`: the path is not a directory.
            - `SandboxFilesystemPermissionError`: read permission is denied.
            - `SandboxFilesystemError`: the command fails for any other reason.

            **Usage**

            ```python fixture:sandbox
            entries = sandbox.filesystem.list_files("/tmp")
            for entry in entries:
                print(entry.name, entry.type, entry.size)
            ```
            """
            ...

    list_files: __list_files_spec

    class __make_directory_spec(typing_extensions.Protocol):
        def __call__(self, /, remote_path: str, *, create_parents: bool = True) -> None:
            """Create a new directory in the Sandbox.

            `remote_path` must be an absolute path in the Sandbox.

            When `create_parents` is `True` (the default), any missing parent directories are created and the call is
            idempotent (succeeds silently if the directory already exists). When `create_parents` is `False`, the
            immediate parent directory must already exist and the path must not already exist.

            **Raises**

            - `SandboxFilesystemNotFoundError`: the parent directory does not exist and `create_parents` is `False`.
            - `SandboxFilesystemPathAlreadyExistsError`: the path already exists.
            - `SandboxFilesystemNotADirectoryError`: a path component is not a directory.
            - `SandboxFilesystemPermissionError`: creation is not permitted.
            - `InvalidError`: the operation is not supported by the mount.
            - `SandboxFilesystemError`: the command fails for any other reason.

            **Usage**

            ```python fixture:sandbox
            sandbox.filesystem.make_directory("/tmp/a/b/c")
            ```
            """
            ...

        async def aio(self, /, remote_path: str, *, create_parents: bool = True) -> None:
            """Create a new directory in the Sandbox.

            `remote_path` must be an absolute path in the Sandbox.

            When `create_parents` is `True` (the default), any missing parent directories are created and the call is
            idempotent (succeeds silently if the directory already exists). When `create_parents` is `False`, the
            immediate parent directory must already exist and the path must not already exist.

            **Raises**

            - `SandboxFilesystemNotFoundError`: the parent directory does not exist and `create_parents` is `False`.
            - `SandboxFilesystemPathAlreadyExistsError`: the path already exists.
            - `SandboxFilesystemNotADirectoryError`: a path component is not a directory.
            - `SandboxFilesystemPermissionError`: creation is not permitted.
            - `InvalidError`: the operation is not supported by the mount.
            - `SandboxFilesystemError`: the command fails for any other reason.

            **Usage**

            ```python fixture:sandbox
            sandbox.filesystem.make_directory("/tmp/a/b/c")
            ```
            """
            ...

    make_directory: __make_directory_spec

    class __read_bytes_spec(typing_extensions.Protocol):
        def __call__(self, /, remote_path: str) -> bytes:
            """Read a file from the Sandbox and return its contents as bytes.

            `remote_path` must be an absolute path to a file in the Sandbox.

            **Raises**

            - `SandboxFilesystemNotFoundError`: the path does not exist.
            - `SandboxFilesystemIsADirectoryError`: the path points to a directory.
            - `SandboxFilesystemPermissionError`: read permission is denied.
            - `SandboxFilesystemError`: the command fails for any other reason.

            **Usage**

            ```python fixture:sandbox
            sandbox.filesystem.write_bytes(b"Hello, world!\n", "/tmp/hello.bin")
            contents = sandbox.filesystem.read_bytes("/tmp/hello.bin")
            print(contents.decode("utf-8"))
            ```
            """
            ...

        async def aio(self, /, remote_path: str) -> bytes:
            """Read a file from the Sandbox and return its contents as bytes.

            `remote_path` must be an absolute path to a file in the Sandbox.

            **Raises**

            - `SandboxFilesystemNotFoundError`: the path does not exist.
            - `SandboxFilesystemIsADirectoryError`: the path points to a directory.
            - `SandboxFilesystemPermissionError`: read permission is denied.
            - `SandboxFilesystemError`: the command fails for any other reason.

            **Usage**

            ```python fixture:sandbox
            sandbox.filesystem.write_bytes(b"Hello, world!\n", "/tmp/hello.bin")
            contents = sandbox.filesystem.read_bytes("/tmp/hello.bin")
            print(contents.decode("utf-8"))
            ```
            """
            ...

    read_bytes: __read_bytes_spec

    class __read_text_spec(typing_extensions.Protocol):
        def __call__(self, /, remote_path: str) -> str:
            """Read a file from the Sandbox and return its contents as a UTF-8 string.

            `remote_path` must be an absolute path to a file in the Sandbox.

            **Raises**

            - `SandboxFilesystemNotFoundError`: the path does not exist.
            - `SandboxFilesystemIsADirectoryError`: the path points to a directory.
            - `SandboxFilesystemPermissionError`: read permission is denied.
            - `SandboxFilesystemError`: the command fails for any other reason.

            **Usage**

            ```python fixture:sandbox
            sandbox.filesystem.write_text("Hello, world!\n", "/tmp/hello.txt")
            contents = sandbox.filesystem.read_text("/tmp/hello.txt")
            print(contents)
            ```
            """
            ...

        async def aio(self, /, remote_path: str) -> str:
            """Read a file from the Sandbox and return its contents as a UTF-8 string.

            `remote_path` must be an absolute path to a file in the Sandbox.

            **Raises**

            - `SandboxFilesystemNotFoundError`: the path does not exist.
            - `SandboxFilesystemIsADirectoryError`: the path points to a directory.
            - `SandboxFilesystemPermissionError`: read permission is denied.
            - `SandboxFilesystemError`: the command fails for any other reason.

            **Usage**

            ```python fixture:sandbox
            sandbox.filesystem.write_text("Hello, world!\n", "/tmp/hello.txt")
            contents = sandbox.filesystem.read_text("/tmp/hello.txt")
            print(contents)
            ```
            """
            ...

    read_text: __read_text_spec

    class __remove_spec(typing_extensions.Protocol):
        def __call__(self, /, remote_path: str, *, recursive: bool = False) -> None:
            """Remove a file or directory in the Sandbox.

            `remote_path` must be an absolute path in the Sandbox.

            When `remote_path` is a directory and `recursive` is `False` (the
            default), removes it only if it is empty. When `recursive` is `True`,
            removes the directory and all its contents.

            Recursive directory removal is not supported on all mounts.
            In particular, `CloudBucketMount` does not support it. An
            `InvalidError` is raised in that case.

            **Raises**

            - `SandboxFilesystemNotFoundError`: the path does not exist.
            - `SandboxFilesystemDirectoryNotEmptyError`: `recursive` is `False` and the directory is not empty.
            - `SandboxFilesystemPermissionError`: removal is not permitted.
            - `InvalidError`: the operation is not supported by the mount.
            - `SandboxFilesystemError`: the command fails for any other reason.

            **Usage**

            To remove a file:

            ```python fixture:sandbox
            sandbox.filesystem.write_bytes(b"Hello, world!\n", "/tmp/hello.bin")
            sandbox.filesystem.remove("/tmp/hello.bin")
            ```

            To remove a directory and all its contents:

            ```python fixture:sandbox
            sandbox.filesystem.make_directory("/tmp/mydir/subdir")
            sandbox.filesystem.remove("/tmp/mydir", recursive=True)
            ```
            """
            ...

        async def aio(self, /, remote_path: str, *, recursive: bool = False) -> None:
            """Remove a file or directory in the Sandbox.

            `remote_path` must be an absolute path in the Sandbox.

            When `remote_path` is a directory and `recursive` is `False` (the
            default), removes it only if it is empty. When `recursive` is `True`,
            removes the directory and all its contents.

            Recursive directory removal is not supported on all mounts.
            In particular, `CloudBucketMount` does not support it. An
            `InvalidError` is raised in that case.

            **Raises**

            - `SandboxFilesystemNotFoundError`: the path does not exist.
            - `SandboxFilesystemDirectoryNotEmptyError`: `recursive` is `False` and the directory is not empty.
            - `SandboxFilesystemPermissionError`: removal is not permitted.
            - `InvalidError`: the operation is not supported by the mount.
            - `SandboxFilesystemError`: the command fails for any other reason.

            **Usage**

            To remove a file:

            ```python fixture:sandbox
            sandbox.filesystem.write_bytes(b"Hello, world!\n", "/tmp/hello.bin")
            sandbox.filesystem.remove("/tmp/hello.bin")
            ```

            To remove a directory and all its contents:

            ```python fixture:sandbox
            sandbox.filesystem.make_directory("/tmp/mydir/subdir")
            sandbox.filesystem.remove("/tmp/mydir", recursive=True)
            ```
            """
            ...

    remove: __remove_spec

    class __stat_spec(typing_extensions.Protocol):
        def __call__(self, /, remote_path: str) -> FileInfo:
            """Return metadata for a single file, directory, or symlink in the Sandbox.

            `remote_path` must be an absolute path in the Sandbox. If `remote_path` is a symlink, the returned
            `FileInfo` object describes the symlink, not the target it points to.

            **Raises**

            - `SandboxFilesystemNotFoundError`: the path does not exist.
            - `SandboxFilesystemNotADirectoryError`: a non-leaf component of the path is not a directory.
            - `SandboxFilesystemPermissionError`: a component of the path is not searchable.
            - `SandboxFilesystemError`: the command fails for any other reason.

            **Usage**

            ```python fixture:sandbox
            sandbox.filesystem.write_text("Hello, world!\n", "/tmp/hello.txt")
            info = sandbox.filesystem.stat("/tmp/hello.txt")
            print(info.size, info.permissions, info.modified_time)
            ```
            """
            ...

        async def aio(self, /, remote_path: str) -> FileInfo:
            """Return metadata for a single file, directory, or symlink in the Sandbox.

            `remote_path` must be an absolute path in the Sandbox. If `remote_path` is a symlink, the returned
            `FileInfo` object describes the symlink, not the target it points to.

            **Raises**

            - `SandboxFilesystemNotFoundError`: the path does not exist.
            - `SandboxFilesystemNotADirectoryError`: a non-leaf component of the path is not a directory.
            - `SandboxFilesystemPermissionError`: a component of the path is not searchable.
            - `SandboxFilesystemError`: the command fails for any other reason.

            **Usage**

            ```python fixture:sandbox
            sandbox.filesystem.write_text("Hello, world!\n", "/tmp/hello.txt")
            info = sandbox.filesystem.stat("/tmp/hello.txt")
            print(info.size, info.permissions, info.modified_time)
            ```
            """
            ...

    stat: __stat_spec

    class __write_bytes_spec(typing_extensions.Protocol):
        def __call__(self, /, data: typing.Union[bytes, bytearray, memoryview], remote_path: str) -> None:
            """Write binary content to a file in the Sandbox.

            `remote_path` must be an absolute path to a file in the Sandbox.
            Parent directories for `remote_path` are created if needed.
            The remote file is overwritten if it already exists.

            **Raises**

            - `SandboxFilesystemNotADirectoryError`: a parent path component is not a directory.
            - `SandboxFilesystemIsADirectoryError`: `remote_path` points to a directory.
            - `SandboxFilesystemPermissionError`: write permission is denied.
            - `SandboxFilesystemError`: the command fails for any other reason.

            **Usage**

            ```python fixture:sandbox
            sandbox.filesystem.write_bytes(b"Hello, world!\n", "/tmp/hello.bin")
            ```
            """
            ...

        async def aio(self, /, data: typing.Union[bytes, bytearray, memoryview], remote_path: str) -> None:
            """Write binary content to a file in the Sandbox.

            `remote_path` must be an absolute path to a file in the Sandbox.
            Parent directories for `remote_path` are created if needed.
            The remote file is overwritten if it already exists.

            **Raises**

            - `SandboxFilesystemNotADirectoryError`: a parent path component is not a directory.
            - `SandboxFilesystemIsADirectoryError`: `remote_path` points to a directory.
            - `SandboxFilesystemPermissionError`: write permission is denied.
            - `SandboxFilesystemError`: the command fails for any other reason.

            **Usage**

            ```python fixture:sandbox
            sandbox.filesystem.write_bytes(b"Hello, world!\n", "/tmp/hello.bin")
            ```
            """
            ...

    write_bytes: __write_bytes_spec

    class __write_text_spec(typing_extensions.Protocol):
        def __call__(self, /, data: str, remote_path: str) -> None:
            """Write UTF-8 text to a file in the Sandbox.

            `remote_path` must be an absolute path to a file in the Sandbox.
            Parent directories for `remote_path` are created if needed.
            The remote file is overwritten if it already exists.

            **Raises**

            - `SandboxFilesystemNotADirectoryError`: a parent path component is not a directory.
            - `SandboxFilesystemIsADirectoryError`: `remote_path` points to a directory.
            - `SandboxFilesystemPermissionError`: write permission is denied.
            - `SandboxFilesystemError`: the command fails for any other reason.

            **Usage**

            ```python fixture:sandbox
            sandbox.filesystem.write_text("Hello, world!\n", "/tmp/hello.txt")
            ```
            """
            ...

        async def aio(self, /, data: str, remote_path: str) -> None:
            """Write UTF-8 text to a file in the Sandbox.

            `remote_path` must be an absolute path to a file in the Sandbox.
            Parent directories for `remote_path` are created if needed.
            The remote file is overwritten if it already exists.

            **Raises**

            - `SandboxFilesystemNotADirectoryError`: a parent path component is not a directory.
            - `SandboxFilesystemIsADirectoryError`: `remote_path` points to a directory.
            - `SandboxFilesystemPermissionError`: write permission is denied.
            - `SandboxFilesystemError`: the command fails for any other reason.

            **Usage**

            ```python fixture:sandbox
            sandbox.filesystem.write_text("Hello, world!\n", "/tmp/hello.txt")
            ```
            """
            ...

    write_text: __write_text_spec
