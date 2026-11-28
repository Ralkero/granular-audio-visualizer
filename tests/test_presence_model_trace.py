"""A5.2 Step 2a, S2a.7: the live-trace reader's rules on hand-written lines in strace's format
(protocol Amendment 3, A3.4). These test the reader's logic only; the real format is exercised by
the sandbox tests, which run strace."""

from gmv.character.research.presence_model import trace

START = [
    '10 execve("/usr/bin/bwrap", [...], 0x7ffd /* 9 vars */) = 0',
    '10 openat(AT_FDCWD, "/home/someone/inputs/manifest.json", O_RDONLY|O_PATH) = 5',  # set-up: not checked
    '11 execve("/usr/bin/python3.14", [...], 0x55 /* 2 vars */) = 0',
]
END = ["11 exit_group(0) = ?", "11 +++ exited with 0 +++", "10 +++ exited with 0 +++"]
INPUTS = ["/in/corpus/manifest.json"]


def verdict(*lines, strace_exit=0, end=True):
    return trace.read("\n".join([*START, *lines, *(END if end else [])]), INPUTS, strace_exit)


def test_set_up_is_kept_apart_and_a_clean_step_is_accepted():
    v = verdict('11 openat(AT_FDCWD, "/gmv/src/gmv/__init__.py", O_RDONLY|O_CLOEXEC) = 3',
                '11 openat(AT_FDCWD, "/in/corpus/manifest.json", O_RDONLY|O_CLOEXEC) = 3',
                '11 openat(AT_FDCWD, "/out/public.json", O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC, 0666) = 4',
                '11 openat(AT_FDCWD, "/usr/pyvenv.cfg", O_RDONLY) = -1 ENOENT (No such file or directory)')
    assert v.accepted and v.counts["set-up"] == 2 and v.counts["input:open"] == 1 and v.counts["out:write"] == 1
    assert v.public()["verdict"] == "accepted"


def test_paths_outside_the_allowed_classes_are_violations_even_when_they_fail():
    v = verdict('11 openat(AT_FDCWD, "/in/presence/identity/groups.json", O_RDONLY) = -1 ENOENT (No such file)',
                '11 newfstatat(AT_FDCWD, "/home/someone", 0x7ff, 0) = -1 ENOENT (No such file)')
    assert not v.accepted and v.violations["open outside the allowed classes"] == 1
    assert v.violations["stat outside the allowed classes"] == 1


def test_a_listing_of_an_input_folder_is_a_violation():
    v = verdict('11 openat(AT_FDCWD, "/in/corpus", O_RDONLY|O_DIRECTORY) = 3',
                '11 getdents64(3, 0x55 /* 3 entries */, 32768) = 80')
    assert v.violations["list outside the allowed classes"] == 1


def test_descriptor_relative_paths_resolve_and_unknown_descriptors_fail_closed():
    ok = verdict('11 openat(AT_FDCWD, "/gmv/src", O_RDONLY|O_DIRECTORY) = 7',
                 '11 newfstatat(7, "gmv/models.py", 0x7ff, 0) = 0',
                 '11 statx(7, "", AT_EMPTY_PATH, STATX_ALL, 0x7ff) = 0')
    assert ok.accepted and ok.counts["runtime:stat"] == 2
    unknown = verdict('11 newfstatat(9, "anything", 0x7ff, 0) = 0')
    assert not unknown.accepted and unknown.unresolved == 1


def test_relative_paths_resolve_against_the_working_folder_and_dot_dot_is_followed():
    v = verdict('11 openat(AT_FDCWD, "gmv/src/../../in/corpus/manifest.json", O_RDONLY) = 3')
    assert v.accepted and v.counts["input:open"] == 1
    escape = verdict('11 openat(AT_FDCWD, "/gmv/src/../../etc/hostname", O_RDONLY) = -1 ENOENT (x)')
    assert not escape.accepted


def test_split_calls_are_joined():
    v = verdict('11 openat(AT_FDCWD, "/in/corpus/manifest.json", O_RDONLY <unfinished ...>',
                '11 <... openat resumed>) = 3')
    assert v.accepted and v.counts["input:open"] == 1
    bad = verdict('11 <... openat resumed>) = 3')
    assert not bad.complete


def test_threads_are_allowed_but_processes_programs_network_moves_and_io_uring_are_not():
    assert verdict('11 clone3({flags=CLONE_VM|CLONE_FS|CLONE_THREAD|CLONE_SIGHAND, ...}, 88) = 12',
                   "12 +++ exited with 0 +++").accepted
    for line, kind in (('11 clone(child_stack=NULL, flags=SIGCHLD) = 13', "a child process"),
                       ('11 execve("/bin/sh", [...], 0x55 /* 2 vars */) = -1 ENOENT (x)', "a new program"),
                       ('11 socket(AF_INET, SOCK_STREAM|SOCK_CLOEXEC, IPPROTO_IP) = 3', "a network call"),
                       ('11 chdir("/out") = 0', "a change of directory or root"),
                       ('11 io_uring_setup(1, 0x7ff) = 3', "io_uring")):
        extra = ["13 +++ exited with 0 +++"] if "13" in line else []
        v = verdict(line, *extra)
        assert v.violations[kind] == 1 and not v.accepted, kind


def test_an_incomplete_trace_is_rejected():
    assert not verdict(end=False).complete  # no exit lines
    assert not verdict(strace_exit=1).complete
    assert not trace.read("\n".join(START[:2] + ["10 +++ exited with 0 +++"]), INPUTS, 0).complete  # no step
    assert not verdict("garbage line").complete
    assert not verdict('11 openat(AT_FDCWD, "/in/corpus/manifest.json", O_RDONLY <unfinished ...>').complete


def test_a_core_dump_is_noticed():
    v = trace.read("\n".join([*START, "11 +++ killed by SIGABRT (core dumped) +++", "10 +++ exited with 1 +++"]),
                   INPUTS, 1)
    assert v.core_dumped and not v.accepted


def test_a_truncated_or_missing_path_is_unresolved():
    v = verdict('11 openat(AT_FDCWD, "/in/corpus/manif"..., O_RDONLY) = 3')
    assert v.unresolved == 1 and not v.accepted


def test_the_public_summary_holds_counts_only():
    v = verdict('11 openat(AT_FDCWD, "/in/corpus/manifest.json", O_RDONLY) = 3')
    text = repr(v.public())
    assert "/in" not in text and "manifest" not in text and "someone" not in text
