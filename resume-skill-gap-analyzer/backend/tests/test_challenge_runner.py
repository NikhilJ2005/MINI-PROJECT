"""
Tests for the function-only code challenge runner (challenge_runner.py).

Covers:
  - Python accepted / wrong answer for Valid Parentheses
  - Java compile error case
  - Response schema keys always present
  - C/C++ accepted case + failed_cases details
  - JS deep equality for arrays/objects
  - JS error classification (Compile Error vs Runtime Error)
  - Template key normalization in problem_detail
"""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Import challenge_runner directly to avoid the modules/__init__.py chain
# (which requires optional heavy dependencies like PyMuPDF)
_spec = importlib.util.spec_from_file_location(
    "challenge_runner",
    BACKEND_DIR / "modules" / "challenge_runner.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

run_challenge = _mod.run_challenge
load_problem = _mod.load_problem
list_problems = _mod.list_problems
problem_detail = _mod.problem_detail
_build_js_harness = _mod._build_js_harness
_find_node = _mod._find_node
_run_subprocess = _mod._run_subprocess

PROBLEM_ID = "valid_parentheses"

REQUIRED_KEYS = {"ok", "verdict", "passed", "total", "runtime_ms",
                 "failed_cases", "stdout", "stderr"}

VALID_VERDICTS = {"Accepted", "Wrong Answer", "Compile Error",
                  "Runtime Error", "Time Limit"}


# -----------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------

def assert_schema(result: dict):
    """Verify that every required key is present in the result dict."""
    for key in REQUIRED_KEYS:
        assert key in result, f"Missing key '{key}' in result: {result}"
    assert result["verdict"] in VALID_VERDICTS, \
        f"Unexpected verdict: {result['verdict']}"
    assert isinstance(result["ok"], bool)
    assert isinstance(result["passed"], int)
    assert isinstance(result["total"], int)
    assert isinstance(result["runtime_ms"], (int, float))
    assert isinstance(result["failed_cases"], list)
    assert isinstance(result["stdout"], str)
    assert isinstance(result["stderr"], str)


# -----------------------------------------------------------------------
# Problem metadata tests
# -----------------------------------------------------------------------

class TestProblemMetadata:
    def test_load_problem(self):
        prob = load_problem(PROBLEM_ID)
        assert prob["id"] == PROBLEM_ID
        assert "title" in prob
        assert "function_name" in prob
        assert "tests" in prob

    def test_list_problems(self):
        probs = list_problems()
        assert isinstance(probs, list)
        assert len(probs) >= 1
        ids = [p["id"] for p in probs]
        assert PROBLEM_ID in ids

    def test_problem_detail_hides_hidden_tests(self):
        detail = problem_detail(PROBLEM_ID)
        # detail must not contain the hidden test set
        assert "hidden" not in str(detail.get("tests", ""))
        assert "sample_tests" in detail
        # starter templates should be present
        assert "starter_templates" in detail

    def test_problem_detail_normalizes_template_keys(self):
        detail = problem_detail(PROBLEM_ID)
        templates = detail["starter_templates"]
        # Frontend uses these exact keys; none of the alias forms should appear.
        for alias in ("js", "c++", "py"):
            assert alias not in templates, f"Alias key '{alias}' should be normalized"
        # Canonical keys should be present (problem has them already, normalization kept them)
        assert "python" in templates
        assert "javascript" in templates

    def test_load_nonexistent_problem(self):
        with pytest.raises(FileNotFoundError):
            load_problem("nonexistent_problem_xyz")


# -----------------------------------------------------------------------
# Python runner
# -----------------------------------------------------------------------

PYTHON_ACCEPTED = """\
def isValid(s: str) -> bool:
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for ch in s:
        if ch in mapping:
            top = stack.pop() if stack else '#'
            if mapping[ch] != top:
                return False
        else:
            stack.append(ch)
    return not stack
"""

PYTHON_WRONG = """\
def isValid(s: str) -> bool:
    return True
"""

PYTHON_RUNTIME_ERROR = """\
def isValid(s: str) -> bool:
    raise RuntimeError("intentional error")
"""

PYTHON_SYNTAX_ERROR = """\
def isValid(s: str) -> bool:
    return (
"""


class TestPythonRunner:
    def test_accepted(self):
        result = run_challenge(PROBLEM_ID, "python", PYTHON_ACCEPTED, mode="sample")
        assert_schema(result)
        assert result["verdict"] == "Accepted"
        assert result["ok"] is True
        assert result["passed"] == result["total"]
        assert result["total"] > 0

    def test_wrong_answer(self):
        result = run_challenge(PROBLEM_ID, "python", PYTHON_WRONG, mode="sample")
        assert_schema(result)
        assert result["verdict"] == "Wrong Answer"
        assert result["ok"] is False
        assert result["passed"] < result["total"]

    def test_runtime_error(self):
        result = run_challenge(PROBLEM_ID, "python", PYTHON_RUNTIME_ERROR, mode="sample")
        assert_schema(result)
        assert result["verdict"] in ("Runtime Error", "Wrong Answer")
        assert result["ok"] is False

    def test_syntax_error(self):
        result = run_challenge(PROBLEM_ID, "python", PYTHON_SYNTAX_ERROR, mode="sample")
        assert_schema(result)
        assert result["ok"] is False

    def test_all_mode_more_tests(self):
        result_sample = run_challenge(PROBLEM_ID, "python", PYTHON_ACCEPTED, mode="sample")
        result_all = run_challenge(PROBLEM_ID, "python", PYTHON_ACCEPTED, mode="all")
        assert_schema(result_all)
        assert result_all["total"] >= result_sample["total"]
        assert result_all["verdict"] == "Accepted"


# -----------------------------------------------------------------------
# JavaScript runner
# -----------------------------------------------------------------------

JS_ACCEPTED = """\
function isValid(s) {
    const stack = [];
    const map = {')': '(', '}': '{', ']': '['};
    for (const ch of s) {
        if (map[ch]) {
            if (stack.pop() !== map[ch]) return false;
        } else {
            stack.push(ch);
        }
    }
    return stack.length === 0;
}
"""

JS_WRONG = """\
function isValid(s) { return true; }
"""


JS_SYNTAX_ERROR = """\
function isValid(s {
    return true;
}
"""

JS_RUNTIME_ERROR = """\
function isValid(s) {
    throw new Error("intentional runtime error");
}
"""


class TestJavaScriptRunner:
    def test_accepted(self):
        result = run_challenge(PROBLEM_ID, "javascript", JS_ACCEPTED, mode="sample")
        assert_schema(result)
        assert result["verdict"] == "Accepted"
        assert result["ok"] is True

    def test_wrong_answer(self):
        result = run_challenge(PROBLEM_ID, "javascript", JS_WRONG, mode="sample")
        assert_schema(result)
        assert result["verdict"] == "Wrong Answer"
        assert result["ok"] is False

    def test_syntax_error_is_compile_error(self):
        result = run_challenge(PROBLEM_ID, "javascript", JS_SYNTAX_ERROR, mode="sample")
        assert_schema(result)
        assert result["verdict"] == "Compile Error"
        assert result["ok"] is False

    def test_runtime_error_is_not_compile_error(self):
        result = run_challenge(PROBLEM_ID, "javascript", JS_RUNTIME_ERROR, mode="sample")
        assert_schema(result)
        # A thrown runtime error must NOT be classified as Compile Error
        assert result["verdict"] != "Compile Error"
        assert result["ok"] is False


# -----------------------------------------------------------------------
# Java runner — mainly tests compile error path
# -----------------------------------------------------------------------

JAVA_COMPILE_ERROR = """\
class Solution {
    public boolean isValid(String s) {
        this is not valid java code
    }
}
"""

JAVA_ACCEPTED = """\
import java.util.Stack;

class Solution {
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for (char c : s.toCharArray()) {
            if (c == '(' || c == '{' || c == '[') {
                stack.push(c);
            } else {
                if (stack.isEmpty()) return false;
                char top = stack.pop();
                if ((c == ')' && top != '(') ||
                    (c == '}' && top != '{') ||
                    (c == ']' && top != '[')) return false;
            }
        }
        return stack.isEmpty();
    }
}
"""


class TestJavaRunner:
    def test_compile_error(self):
        result = run_challenge(PROBLEM_ID, "java", JAVA_COMPILE_ERROR, mode="sample")
        assert_schema(result)
        assert result["verdict"] == "Compile Error"
        assert result["ok"] is False
        assert result["passed"] == 0
        # stderr should contain error message
        assert len(result["stderr"]) > 0

    def test_accepted(self):
        result = run_challenge(PROBLEM_ID, "java", JAVA_ACCEPTED, mode="sample")
        assert_schema(result)
        assert result["verdict"] == "Accepted"
        assert result["ok"] is True


# -----------------------------------------------------------------------
# C runner
# -----------------------------------------------------------------------

C_ACCEPTED = """\
#include <string.h>

int isValid(const char* s) {
    char stack[10001];
    int top = 0;
    for (int i = 0; s[i]; i++) {
        char c = s[i];
        if (c == '(' || c == '{' || c == '[') {
            stack[top++] = c;
        } else {
            if (top == 0) return 0;
            char t = stack[--top];
            if ((c == ')' && t != '(') ||
                (c == '}' && t != '{') ||
                (c == ']' && t != '[')) return 0;
        }
    }
    return top == 0;
}
"""

C_COMPILE_ERROR = """\
int isValid(const char* s) {
    this is not valid C;
}
"""


C_WRONG = """\
int isValid(const char* s) {
    return 0;  /* always false */
}
"""


class TestCRunner:
    def test_accepted(self):
        result = run_challenge(PROBLEM_ID, "c", C_ACCEPTED, mode="sample")
        assert_schema(result)
        assert result["verdict"] == "Accepted"
        assert result["ok"] is True

    def test_compile_error(self):
        result = run_challenge(PROBLEM_ID, "c", C_COMPILE_ERROR, mode="sample")
        assert_schema(result)
        assert result["verdict"] == "Compile Error"
        assert result["ok"] is False

    def test_wrong_answer_returns_failed_cases(self):
        result = run_challenge(PROBLEM_ID, "c", C_WRONG, mode="sample")
        assert_schema(result)
        assert result["verdict"] == "Wrong Answer"
        assert result["ok"] is False
        # At least one failed case must be reported with details
        assert len(result["failed_cases"]) > 0
        fc = result["failed_cases"][0]
        assert "index" in fc
        assert "input" in fc
        assert "expected" in fc


# -----------------------------------------------------------------------
# C++ runner
# -----------------------------------------------------------------------

CPP_ACCEPTED = """\
#include <string>
#include <stack>
using namespace std;

bool isValid(string s) {
    stack<char> st;
    for (char c : s) {
        if (c == '(' || c == '{' || c == '[') {
            st.push(c);
        } else {
            if (st.empty()) return false;
            char t = st.top(); st.pop();
            if ((c == ')' && t != '(') ||
                (c == '}' && t != '{') ||
                (c == ']' && t != '[')) return false;
        }
    }
    return st.empty();
}
"""


CPP_WRONG = """\
#include <string>
using namespace std;

bool isValid(string s) {
    return false;  // always false
}
"""


class TestCppRunner:
    def test_accepted(self):
        result = run_challenge(PROBLEM_ID, "cpp", CPP_ACCEPTED, mode="sample")
        assert_schema(result)
        assert result["verdict"] == "Accepted"
        assert result["ok"] is True

    def test_wrong_answer_returns_failed_cases(self):
        result = run_challenge(PROBLEM_ID, "cpp", CPP_WRONG, mode="sample")
        assert_schema(result)
        assert result["verdict"] == "Wrong Answer"
        assert result["ok"] is False
        assert len(result["failed_cases"]) > 0
        fc = result["failed_cases"][0]
        assert "index" in fc
        assert "input" in fc
        assert "expected" in fc


# -----------------------------------------------------------------------
# Schema completeness — all languages must return all keys
# -----------------------------------------------------------------------

class TestSchemaCompleteness:
    """Ensure that regardless of verdict, all schema keys are always present."""

    @pytest.mark.parametrize("lang,code", [
        ("python", PYTHON_ACCEPTED),
        ("python", PYTHON_WRONG),
        ("javascript", JS_WRONG),
        ("java", JAVA_COMPILE_ERROR),
        ("c", C_COMPILE_ERROR),
    ])
    def test_schema_keys_present(self, lang, code):
        result = run_challenge(PROBLEM_ID, lang, code, mode="sample")
        assert_schema(result)

    def test_unknown_language_schema(self):
        result = run_challenge(PROBLEM_ID, "brainfuck", "anything", mode="sample")
        assert_schema(result)
        assert result["ok"] is False

    def test_unknown_problem_schema(self):
        result = run_challenge("no_such_problem", "python", "def f(): pass", mode="sample")
        assert_schema(result)
        assert result["ok"] is False

# -----------------------------------------------------------------------
# JS deep equality — arrays and objects must compare by value, not ref
# -----------------------------------------------------------------------

class TestJSDeepEquality:
    """Verify that JS harness uses deep equality so array/object returns work."""

    def test_array_return_accepted(self):
        """A function returning an array must be accepted when values match."""
        # Synthetic problem with array return
        mock_problem = {
            "function_name": "twoSum",
            "returns": "array",
        }
        tests = [
            {"input": [[2, 7, 11, 15], 9], "expected": [0, 1]},
            {"input": [[3, 2, 4], 6],      "expected": [1, 2]},
        ]
        user_code = """\
function twoSum(nums, target) {
    const map = {};
    for (let i = 0; i < nums.length; i++) {
        const comp = target - nums[i];
        if (comp in map) return [map[comp], i];
        map[nums[i]] = i;
    }
}
"""
        harness = _build_js_harness(mock_problem, user_code, tests)
        node_cmd = _find_node()
        with tempfile.TemporaryDirectory() as tmpdir:
            harness_path = os.path.join(tmpdir, "solution.js")
            with open(harness_path, "w") as f:
                f.write(harness)
            res = _run_subprocess([node_cmd, "solution.js"], cwd=tmpdir)
        assert res["returncode"] == 0, f"Node error: {res['stderr']}"
        out = json.loads(res["stdout"].strip())
        assert out["passed"] == 2, f"Expected 2 passed, got: {out}"
        assert out["failed"] == [], f"Expected no failures, got: {out['failed']}"
