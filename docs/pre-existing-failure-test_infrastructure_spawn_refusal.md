# Pre-existing test failure (unrelated to ops-reliability fixes)

## Failure

- Test: `tests/hermes_cli/test_kanban_db.py::test_infrastructure_spawn_refusal_never_charges_the_card`
- Symptom: `assert res.auto_blocked == []` fails — `auto_blocked` is `['t_<task_id>']`.
- Root cause: unrelated to the two fixes on this branch (does not touch `request_review`,
  the SSH-forwarding helpers, or any code path modified here).

## Evidence it reproduces WITHOUT our changes

Confirmed twice this session by stashing out all source changes and running just that test:

1. `git stash push -- hermes_cli/kanban_db.py hermes_cli/config_defaults.py hermes_cli/kanban_db_dispatch.py`
2. `bash scripts/run_tests.sh "tests/hermes_cli/test_kanban_db.py::test_infrastructure_spawn_refusal_never_charges_the_card" -q`

Result both times (clean upstream tree):
```
E   AssertionError: assert ['t_<id>'] == []
E      Left contains one more item: 't_<id>'
tests/hermes_cli/test_kanban_db.py:578: AssertionError
1 failed, 54 deselected
```

The same single failure is present with the fixes applied:
```
1 failed, 53 passed, 1 skipped   (test_kanban_db.py + test_kanban_db_repair.py + test_kanban_db_dispatch.py)
```

The failing test exercises the "bus is down" infra-refusal spawn path with a mocked
`spawn_via_real_boundary`; it is independent of dispatcher-recovery and SSH-forwarding logic.

## Independent reviewer's note

The reviewer did not re-stash to reconfirm, but independently verified that the failing
test "does not touch any code path in this change."