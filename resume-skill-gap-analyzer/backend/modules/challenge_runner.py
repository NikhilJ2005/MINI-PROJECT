"""
challenge_runner.py — Multi-language function-only code challenge runner.

Loads a problem spec by id, generates a per-language harness/wrapper,
executes in a temp dir with strict timeouts, compares outputs, and
returns a unified result schema.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROBLEMS_DIR = Path(__file__).resolve().parent.parent / "problems"
TIMEOUT_SECONDS = 10  # per-run wall-clock limit
MAX_OUTPUT_BYTES = 1 * 1024 * 1024  # 1 MiB per stream

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result schema helpers
# ---------------------------------------------------------------------------

def _ok_result(passed: int, total: int, runtime_ms: float,
                failed_cases: List[Dict], stdout: str, stderr: str) -> Dict:
    verdict = "Accepted" if passed == total else "Wrong Answer"
    return {
        "ok": passed == total,
        "verdict": verdict,
        "passed": passed,
        "total": total,
        "runtime_ms": round(runtime_ms, 2),
        "failed_cases": failed_cases,
        "stdout": stdout,
        "stderr": stderr,
    }


def _error_result(verdict: str, stderr: str, stdout: str = "",
                  total: int = 0) -> Dict:
    return {
        "ok": False,
        "verdict": verdict,
        "passed": 0,
        "total": total,
        "runtime_ms": 0.0,
        "failed_cases": [],
        "stdout": stdout,
        "stderr": stderr,
    }


# ---------------------------------------------------------------------------
# Problem loading
# ---------------------------------------------------------------------------

def load_problem(problem_id: str) -> Dict:
    """Load a problem JSON from the problems directory."""
    path = PROBLEMS_DIR / f"{problem_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Problem '{problem_id}' not found.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_problems() -> List[Dict]:
    """Return a list of all problems (id, title, difficulty) without exposing tests."""
    problems = []
    for p in sorted(PROBLEMS_DIR.glob("*.json")):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        problems.append({
            "id": data["id"],
            "title": data["title"],
            "difficulty": data["difficulty"],
        })
    return problems


_TEMPLATE_ALIASES: Dict[str, str] = {
    "js": "javascript",
    "c++": "cpp",
    "py": "python",
}

# Frontend language values that must always be present.
_FRONTEND_LANGUAGES = ("python", "javascript", "java", "c", "cpp")


def problem_detail(problem_id: str) -> Dict:
    """Return problem detail without exposing hidden tests."""
    data = load_problem(problem_id)

    # Normalize starter_template keys so frontend language values always resolve.
    raw_templates: Dict[str, str] = data.get("starter_templates", {})
    templates: Dict[str, str] = {}
    for key, val in raw_templates.items():
        canonical = _TEMPLATE_ALIASES.get(key.lower(), key.lower())
        templates[canonical] = val

    return {
        "id": data["id"],
        "title": data["title"],
        "difficulty": data["difficulty"],
        "prompt": data["prompt"],
        "constraints": data.get("constraints", []),
        "function_name": data["function_name"],
        "params": data["params"],
        "returns": data["returns"],
        "examples": data.get("examples", []),
        "starter_templates": templates,
        # Only expose sample tests (not hidden)
        "sample_tests": data.get("tests", {}).get("sample", []),
    }


# ---------------------------------------------------------------------------
# Value serialisation helpers (used in harnesses)
# ---------------------------------------------------------------------------

def _py_literal(value: Any) -> str:
    """Convert a Python value to a Python literal string."""
    return repr(value)


def _js_literal(value: Any) -> str:
    """Convert a Python value to a JSON/JS literal."""
    return json.dumps(value)


def _escape_c_string(s: str) -> str:
    """Escape a Python string for embedding in a C/C++ string literal."""
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    s = s.replace("\t", "\\t")
    return s


def _c_literal(value: Any) -> str:
    """Convert a value to a C/C++ literal (supports bool, string, int, float)."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, str):
        return f'"{_escape_c_string(value)}"'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    raise ValueError(f"Unsupported C literal type: {type(value)}")


def _java_literal(value: Any) -> str:
    """Convert a value to a Java literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        s = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{s}"'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value) + "f"
    raise ValueError(f"Unsupported Java literal type: {type(value)}")


def _returns_to_c_type(returns: str) -> str:
    mapping = {"bool": "int", "int": "int", "float": "double",
               "double": "double", "string": "const char*", "str": "const char*"}
    return mapping.get(returns, "int")


def _returns_to_cpp_type(returns: str) -> str:
    mapping = {"bool": "bool", "int": "int", "float": "double",
               "double": "double", "string": "std::string", "str": "std::string"}
    return mapping.get(returns, "bool")


def _returns_to_java_type(returns: str) -> str:
    mapping = {"bool": "boolean", "int": "int", "float": "float",
               "double": "double", "string": "String", "str": "String"}
    return mapping.get(returns, "boolean")



# ---------------------------------------------------------------------------
# Harness builders — one per language
# ---------------------------------------------------------------------------

def _build_python_harness(problem: Dict, user_code: str, tests: List[Dict]) -> str:
    fn = problem["function_name"]
    lines = [user_code, "", "import sys as _sys", ""]
    lines.append("_tests = [")
    for t in tests:
        inp = t["input"]
        exp = t["expected"]
        lines.append(f"    ({_py_literal(inp)}, {_py_literal(exp)}),")
    lines.append("]")
    lines.append("")
    lines.append("_passed = 0")
    lines.append("_failed = []")
    lines.append("for _i, (_inp, _exp) in enumerate(_tests):")
    lines.append(f"    try:")
    lines.append(f"        _actual = {fn}(*_inp)")
    lines.append(f"    except Exception as _e:")
    lines.append(f"        _failed.append((_i, _inp, _exp, None, str(_e)))")
    lines.append(f"        continue")
    lines.append(f"    if _actual == _exp:")
    lines.append(f"        _passed += 1")
    lines.append(f"    else:")
    lines.append(f"        _failed.append((_i, _inp, _exp, _actual, ''))")
    lines.append("")
    lines.append("import json as _json")
    lines.append("print(_json.dumps({'passed': _passed, 'total': len(_tests), 'failed': _failed}))")
    return "\n".join(lines)


def _build_js_harness(problem: Dict, user_code: str, tests: List[Dict]) -> str:
    fn = problem["function_name"]
    test_json = json.dumps(tests)
    harness = f"""{user_code}

const _util = require('util');
const _tests = {test_json};
let _passed = 0;
const _failed = [];

for (let _i = 0; _i < _tests.length; _i++) {{
  const _t = _tests[_i];
  let _actual;
  try {{
    _actual = {fn}(..._t.input);
  }} catch(e) {{
    _failed.push([_i, _t.input, _t.expected, null, e.message]);
    continue;
  }}
  if (_util.isDeepStrictEqual(_actual, _t.expected)) {{
    _passed++;
  }} else {{
    _failed.push([_i, _t.input, _t.expected, _actual, '']);
  }}
}}

console.log(JSON.stringify({{passed: _passed, total: _tests.length, failed: _failed}}));
"""
    return harness


def _build_c_harness(problem: Dict, user_code: str, tests: List[Dict]) -> str:
    """Generate a self-contained C program with test cases embedded as literals."""
    fn = problem["function_name"]
    returns = problem.get("returns", "bool")
    c_type = _returns_to_c_type(returns)

    # Build case arrays
    case_lines = []
    for i, t in enumerate(tests):
        # MVP: single string param
        inp_val = _c_literal(t["input"][0]) if t["input"] else '""'
        exp_val = _c_literal(t["expected"])
        case_lines.append(f'    {{ {inp_val}, {exp_val} }}')

    cases_str = ",\n".join(case_lines)
    n = len(tests)

    source = f"""#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ---- User code ---- */
{user_code}
/* ---- End user code ---- */

typedef struct {{
    const char* input;
    int expected;
}} TestCase;

int main(void) {{
    TestCase cases[] = {{
{cases_str}
    }};
    int total = {n};
    int passed = 0;
    int failed_indices[{max(n, 1)}];
    int failed_count = 0;

    for (int i = 0; i < total; i++) {{
        {c_type} actual = {fn}(cases[i].input);
        int actual_int = (int)actual;
        if (actual_int == cases[i].expected) {{
            passed++;
        }} else {{
            failed_indices[failed_count++] = i;
        }}
    }}

    printf("{{\\\"passed\\\":%d,\\\"total\\\":%d,\\\"failed_count\\\":%d,\\\"failed_indices\\\":[",
           passed, total, failed_count);
    for (int i = 0; i < failed_count; i++) {{
        if (i > 0) printf(",");
        printf("%d", failed_indices[i]);
    }}
    printf("]}}\\n");
    return 0;
}}
"""
    return source


def _build_cpp_harness(problem: Dict, user_code: str, tests: List[Dict]) -> str:
    """Generate a self-contained C++ program with test cases embedded as literals."""
    fn = problem["function_name"]
    returns = problem.get("returns", "bool")

    # Build input/expected vectors
    input_literals = [_c_literal(t["input"][0]) if t["input"] else '""' for t in tests]
    expected_literals = [("true" if t["expected"] else "false") if isinstance(t["expected"], bool)
                        else str(t["expected"]) for t in tests]
    n = len(tests)
    inputs_str = ", ".join(input_literals)
    expected_str = ", ".join(expected_literals)

    source = f"""#include <iostream>
#include <string>
#include <vector>
using namespace std;

/* ---- User code ---- */
{user_code}
/* ---- End user code ---- */

int main() {{
    vector<string> inputs = {{ {inputs_str} }};
    vector<bool> expected = {{ {expected_str} }};
    int total = {n};
    int passed = 0;
    vector<int> failed_idx;

    for (int i = 0; i < total; i++) {{
        bool actual;
        try {{
            actual = {fn}(inputs[i]);
        }} catch (...) {{
            failed_idx.push_back(i);
            continue;
        }}
        if (actual == expected[i]) {{
            passed++;
        }} else {{
            failed_idx.push_back(i);
        }}
    }}

    cout << "{{\\\"passed\\\":" << passed << ",\\\"total\\\":" << total
         << ",\\\"failed_count\\\":" << (int)failed_idx.size()
         << ",\\\"failed_indices\\\":[";
    for (int i = 0; i < (int)failed_idx.size(); i++) {{
        if (i > 0) cout << ",";
        cout << failed_idx[i];
    }}
    cout << "]}}" << endl;
    return 0;
}}
"""
    return source


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _run_subprocess(cmd: List[str], cwd: str, stdin: Optional[str] = None) -> Dict:
    """Run a subprocess with timeout, output-size limit, and process-tree kill."""
    start = time.monotonic()
    kwargs: Dict[str, Any] = {
        "stdin": subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "cwd": cwd,
    }
    # Create a new process group so we can kill the whole tree on timeout.
    if os.name != "nt":
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **kwargs)
    stdin_bytes = stdin.encode() if stdin is not None else None

    try:
        raw_out, raw_err = proc.communicate(input=stdin_bytes, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        # Kill the entire process group on POSIX; fall back to simple kill on Windows.
        if os.name != "nt":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
        else:
            proc.kill()
        proc.communicate()  # drain to avoid zombie
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "Time limit exceeded.",
            "runtime_ms": TIMEOUT_SECONDS * 1000,
            "timed_out": True,
        }

    elapsed = (time.monotonic() - start) * 1000

    # Enforce output size limit to prevent memory exhaustion (byte-accurate check).
    if len(raw_out) > MAX_OUTPUT_BYTES:
        raw_out = raw_out[:MAX_OUTPUT_BYTES]
        stdout_str = raw_out.decode(errors="replace") + "\n[output truncated]"
    else:
        stdout_str = raw_out.decode(errors="replace")

    if len(raw_err) > MAX_OUTPUT_BYTES:
        raw_err = raw_err[:MAX_OUTPUT_BYTES]
        stderr_str = raw_err.decode(errors="replace") + "\n[output truncated]"
    else:
        stderr_str = raw_err.decode(errors="replace")

    return {
        "returncode": proc.returncode,
        "stdout": stdout_str,
        "stderr": stderr_str,
        "runtime_ms": elapsed,
        "timed_out": False,
    }


def _parse_runner_output(raw: str, tests: List[Dict]) -> Dict[str, Any]:
    """Parse JSON output from Python/JS runners."""
    raw = raw.strip()
    # Find the last JSON object in stdout (guards against extra print()s)
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    return {"passed": 0, "total": len(tests), "failed": []}


def _build_failed_cases(runner_failed: list, tests: List[Dict]) -> List[Dict]:
    """Convert runner failed list to unified schema."""
    result = []
    for item in runner_failed:
        if isinstance(item, (list, tuple)) and len(item) >= 4:
            idx, inp, exp, act = item[0], item[1], item[2], item[3]
            err = item[4] if len(item) > 4 else ""
        elif isinstance(item, dict):
            idx = item.get("index", 0)
            inp = tests[idx]["input"] if idx < len(tests) else []
            exp = tests[idx]["expected"] if idx < len(tests) else None
            act = item.get("actual")
            err = item.get("error", "")
        else:
            continue
        result.append({
            "index": idx,
            "input": inp,
            "expected": exp,
            "actual": act,
            "error": err,
        })
    return result


# ---------------------------------------------------------------------------
# Language-specific runners
# ---------------------------------------------------------------------------

def _run_python(problem: Dict, user_code: str, tests: List[Dict],
                tmpdir: str) -> Dict:
    harness = _build_python_harness(problem, user_code, tests)
    harness_path = os.path.join(tmpdir, "solution.py")
    with open(harness_path, "w", encoding="utf-8") as f:
        f.write(harness)

    result = _run_subprocess([sys.executable, "solution.py"], cwd=tmpdir)
    if result["timed_out"]:
        return _error_result("Time Limit", result["stderr"], result["stdout"],
                             total=len(tests))
    if result["returncode"] != 0:
        return _error_result("Runtime Error", result["stderr"], result["stdout"],
                             total=len(tests))

    parsed = _parse_runner_output(result["stdout"], tests)
    failed = _build_failed_cases(parsed.get("failed", []), tests)
    return _ok_result(parsed.get("passed", 0), parsed.get("total", len(tests)),
                      result["runtime_ms"], failed,
                      result["stdout"], result["stderr"])


def _run_javascript(problem: Dict, user_code: str, tests: List[Dict],
                    tmpdir: str) -> Dict:
    harness = _build_js_harness(problem, user_code, tests)
    harness_path = os.path.join(tmpdir, "solution.js")
    with open(harness_path, "w", encoding="utf-8") as f:
        f.write(harness)

    node_cmd = _find_node()
    result = _run_subprocess([node_cmd, "solution.js"], cwd=tmpdir)
    if result["timed_out"]:
        return _error_result("Time Limit", result["stderr"], result["stdout"],
                             total=len(tests))
    if result["returncode"] != 0:
        stderr = result["stderr"]
        # Classify as Compile Error only for actual syntax/parse failures.
        # Node.js uses exit code 1 for both syntax errors and runtime errors,
        # so inspect stderr rather than relying on the exit code alone.
        _COMPILE_ERROR_MARKERS = (
            "SyntaxError",
            "Unexpected token",
            "Unexpected end of input",
            "Invalid or unexpected token",
            "Cannot use import statement",
            "Cannot find module",
        )
        is_compile_error = any(marker in stderr for marker in _COMPILE_ERROR_MARKERS)
        if is_compile_error:
            return _error_result("Compile Error", stderr, result["stdout"],
                                 total=len(tests))
        return _error_result("Runtime Error", stderr, result["stdout"],
                             total=len(tests))

    parsed = _parse_runner_output(result["stdout"], tests)
    failed_raw = parsed.get("failed", [])
    failed = _build_failed_cases(failed_raw, tests)
    return _ok_result(parsed.get("passed", 0), parsed.get("total", len(tests)),
                      result["runtime_ms"], failed,
                      result["stdout"], result["stderr"])


def _find_node() -> str:
    for candidate in ["node", "/home/runner/work/_temp/ghcca-node/node/bin/node"]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
        try:
            subprocess.run([candidate, "--version"], capture_output=True, timeout=3)
            return candidate
        except Exception:
            pass
    return "node"


def _run_java(problem: Dict, user_code: str, tests: List[Dict],
              tmpdir: str) -> Dict:
    source = _build_java_simple_harness(problem, user_code, tests)
    src_path = os.path.join(tmpdir, "Main.java")
    with open(src_path, "w", encoding="utf-8") as f:
        f.write(source)

    compile_result = _run_subprocess(["javac", "Main.java"], cwd=tmpdir)
    if compile_result["returncode"] != 0:
        return _error_result("Compile Error",
                             compile_result["stderr"], compile_result["stdout"],
                             total=len(tests))

    run_result = _run_subprocess(["java", "-cp", ".", "Main"], cwd=tmpdir)
    if run_result["timed_out"]:
        return _error_result("Time Limit", run_result["stderr"], run_result["stdout"],
                             total=len(tests))
    if run_result["returncode"] != 0:
        return _error_result("Runtime Error", run_result["stderr"], run_result["stdout"],
                             total=len(tests))

    parsed = _parse_runner_output(run_result["stdout"], tests)
    failed_count = parsed.get("failed_count", len(tests) - parsed.get("passed", 0))
    # Build minimal failed_cases list from indices
    failed = []
    for idx in parsed.get("failed_indices", []):
        if idx < len(tests):
            t = tests[idx]
            failed.append({
                "index": idx,
                "input": t["input"],
                "expected": t["expected"],
                "actual": None,
                "error": "",
            })

    return _ok_result(parsed.get("passed", 0), parsed.get("total", len(tests)),
                      run_result["runtime_ms"], failed,
                      run_result["stdout"], run_result["stderr"])


def _build_java_simple_harness(problem: Dict, user_code: str, tests: List[Dict]) -> str:
    """Build a Java harness using only stdlib (no org.json)."""
    fn = problem["function_name"]
    returns = problem.get("returns", "bool")
    java_ret = _returns_to_java_type(returns)
    n = len(tests)

    # Build input and expected arrays inline
    input_lines = []
    expected_lines = []
    for t in tests:
        input_lines.append(_java_literal(t["input"][0]) if t["input"] else '""')
        expected_lines.append(_java_literal(t["expected"]))

    inputs_str = ",\n            ".join(input_lines)
    expected_str = ",\n            ".join(expected_lines)

    return f"""
{user_code}

public class Main {{
    public static void main(String[] args) {{
        Solution _sol = new Solution();
        String[] _inputs = {{
            {inputs_str}
        }};
        boolean[] _expected = {{
            {expected_str}
        }};

        int _passed = 0;
        int _failedCount = 0;
        StringBuilder _failedIdx = new StringBuilder();

        for (int _i = 0; _i < {n}; _i++) {{
            boolean _actual;
            try {{
                _actual = _sol.{fn}(_inputs[_i]);
            }} catch (Exception _e) {{
                _failedCount++;
                if (_failedIdx.length() > 0) _failedIdx.append(",");
                _failedIdx.append(_i);
                continue;
            }}
            if (_actual == _expected[_i]) {{
                _passed++;
            }} else {{
                _failedCount++;
                if (_failedIdx.length() > 0) _failedIdx.append(",");
                _failedIdx.append(_i);
            }}
        }}

        System.out.println(
            "{{\\\"passed\\\":" + _passed +
            ",\\\"total\\\":" + {n} +
            ",\\\"failed_count\\\":" + _failedCount +
            ",\\\"failed_indices\\\":[" + _failedIdx + "]}}"
        );
    }}
}}
"""


def _run_c(problem: Dict, user_code: str, tests: List[Dict],
           tmpdir: str) -> Dict:
    harness = _build_c_harness(problem, user_code, tests)
    src_path = os.path.join(tmpdir, "solution.c")
    with open(src_path, "w", encoding="utf-8") as f:
        f.write(harness)

    compile_result = _run_subprocess(
        ["gcc", "-O2", "-o", "solution", "solution.c"], cwd=tmpdir)
    if compile_result["returncode"] != 0:
        return _error_result("Compile Error",
                             compile_result["stderr"], compile_result["stdout"],
                             total=len(tests))

    run_result = _run_subprocess(["./solution"], cwd=tmpdir)
    if run_result["timed_out"]:
        return _error_result("Time Limit", run_result["stderr"], run_result["stdout"],
                             total=len(tests))
    if run_result["returncode"] != 0:
        return _error_result("Runtime Error", run_result["stderr"], run_result["stdout"],
                             total=len(tests))

    parsed = _parse_runner_output(run_result["stdout"], tests)
    passed = parsed.get("passed", 0)
    total = parsed.get("total", len(tests))
    failed = []
    for idx in parsed.get("failed_indices", []):
        if isinstance(idx, int) and idx < len(tests):
            t = tests[idx]
            failed.append({
                "index": idx,
                "input": t["input"],
                "expected": t["expected"],
                "actual": None,
                "error": "",
            })
    return _ok_result(passed, total, run_result["runtime_ms"], failed,
                      run_result["stdout"], run_result["stderr"])


def _run_cpp(problem: Dict, user_code: str, tests: List[Dict],
             tmpdir: str) -> Dict:
    harness = _build_cpp_harness(problem, user_code, tests)
    src_path = os.path.join(tmpdir, "solution.cpp")
    with open(src_path, "w", encoding="utf-8") as f:
        f.write(harness)

    compile_result = _run_subprocess(
        ["g++", "-O2", "-std=c++17", "-o", "solution", "solution.cpp"], cwd=tmpdir)
    if compile_result["returncode"] != 0:
        return _error_result("Compile Error",
                             compile_result["stderr"], compile_result["stdout"],
                             total=len(tests))

    run_result = _run_subprocess(["./solution"], cwd=tmpdir)
    if run_result["timed_out"]:
        return _error_result("Time Limit", run_result["stderr"], run_result["stdout"],
                             total=len(tests))
    if run_result["returncode"] != 0:
        return _error_result("Runtime Error", run_result["stderr"], run_result["stdout"],
                             total=len(tests))

    parsed = _parse_runner_output(run_result["stdout"], tests)
    passed = parsed.get("passed", 0)
    total = parsed.get("total", len(tests))
    failed = []
    for idx in parsed.get("failed_indices", []):
        if isinstance(idx, int) and idx < len(tests):
            t = tests[idx]
            failed.append({
                "index": idx,
                "input": t["input"],
                "expected": t["expected"],
                "actual": None,
                "error": "",
            })
    return _ok_result(passed, total, run_result["runtime_ms"], failed,
                      run_result["stdout"], run_result["stderr"])


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

LANGUAGE_MAP = {
    "python": _run_python,
    "javascript": _run_javascript,
    "js": _run_javascript,
    "java": _run_java,
    "c": _run_c,
    "cpp": _run_cpp,
    "c++": _run_cpp,
}


def run_challenge(problem_id: str, language: str, user_code: str,
                  mode: str = "sample") -> Dict:
    """
    Run a code challenge submission.

    Args:
        problem_id: Problem identifier (e.g. "valid_parentheses")
        language:   One of python, javascript, java, c, cpp/c++
        user_code:  The candidate's function-only code
        mode:       "sample" (sample tests only) or "all" (all tests)

    Returns:
        Unified result dict with keys: ok, verdict, passed, total,
        runtime_ms, failed_cases, stdout, stderr
    """
    try:
        problem = load_problem(problem_id)
    except FileNotFoundError as exc:
        return _error_result("Runtime Error", str(exc))

    lang_key = language.lower().strip()
    runner_fn = LANGUAGE_MAP.get(lang_key)
    if runner_fn is None:
        return _error_result("Runtime Error",
                             f"Unsupported language: {language}")

    tests_data = problem.get("tests", {})
    if mode == "all":
        tests = tests_data.get("sample", []) + tests_data.get("hidden", [])
    else:
        tests = tests_data.get("sample", [])

    if not tests:
        return _error_result("Runtime Error", "No test cases available.")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            return runner_fn(problem, user_code, tests, tmpdir)
        except Exception as exc:  # pragma: no cover
            return _error_result("Runtime Error", str(exc), total=len(tests))
