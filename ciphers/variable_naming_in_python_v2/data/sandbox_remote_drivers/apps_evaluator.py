# ruff: noqa
# Vendored from hendrycks/apps; only comments have been added to upstream code.
# Revision: b45c0ed78517a3a6492eb77b21cffbb79b1096f1; original: eval/testing_util.py.
# Copyright (c) 2021 Dan Hendrycks; MIT license: see adjacent APPS_LICENSE.txt.
# Uploaded as source data to Modal; never import this file in the local loader.
# Reading map: run_test -> choose interface -> load solution -> loop over cases -> verdicts.
# Both modes are used: default reference examples are call-based; the Codex demo uses stdio.
# The orchestrator checks comment-free tokens against upstream before every submission.
# The license is included here so it accompanies copies uploaded to Modal.
# MIT License
# 
# Copyright (c) 2021 Dan Hendrycks
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =========================== SHARED RUNNER SUPPORT ===========================
import argparse  # Used only by the legacy CLI, not by our evaluation path.
import json
import os
import sys
import io  # Unused upstream import, retained for source equivalence.
import faulthandler
import platform

# used for debugging to time steps
from datetime import datetime

# to run the solution files we're using a timing based approach
import signal

import numpy as np
# for capturing the stdout
from io import StringIO
from typing import get_type_hints  # Unused upstream import, retained for equivalence.
from typing import List, Tuple  # List annotates run_test; Tuple is unused by the evaluator.
# used for testing the code that reads from input
from unittest.mock import patch, mock_open

from pyext import RuntimeModule

from enum import Enum
# fn_name selects one of these invocation modes; neither branch is redundant.
class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1

# stuff for setting up signal timer
# SIGALRM raises an exception; the worker overrides the default timeout below.
class TimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    print("alarm went off")
    #return
    raise TimeoutException
signal.signal(signal.SIGALRM, timeout_handler)
timeout = 4  # seconds

# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
# Stdio-only helper: collect solution stdout for comparison, separate from driver logs.
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


# ====================== LEGACY CLI: UNUSED BY OUR WRAPPER =====================
# Our worker calls run_test with in-memory data; it never invokes this argparse CLI.
def parse_args():
    parser = argparse.ArgumentParser(description="Utility for testing code generation.")
    parser.add_argument("-v", "--verbosity-level", action="store", type=int,
            help="")
    parser.add_argument("-s", "--source",  type=str, default="leetcode", 
            choices=["leetcode", "atcoder", "codewars",],
            help="which data source to gather from.")
    parser.add_argument("-d", "--data",  type=str, default="question",
            choices=["question", "q", "solutions", "sol", "s", "starter", "tests", "t"],
            help="which type of data to receive.")
    parser.add_argument("-n", "--number",  type=int, default=0,
            help="which problem to query.")

    args = parser.parse_args()
    return args


# These on-disk dataset helpers are unused here; apps.py loads Hugging Face rows instead.
def get_valid_problems(data_dir="leetcode"):
    # these are unnecessary atm
    if data_dir == "leetcode":
        root = os.path.join(args.source, "data")
    elif data_dir == "atcoder":
        pass

    root = os.path.join(data_dir, "data")
    if os.path.exists(os.path.join(data_dir, "valid_problems.json")):
        with open(os.path.join(data_dir, "valid_problems.json"), "r") as f:
            return json.load(f)

    # after we compute it once let's save it and load that instead
    # TODO determine if might be better to reload each time
    tmp = os.listdir(root)
    valid_probs = []
    for folder in tmp:
        prob_path = os.path.join(root, folder)
        files = os.listdir(prob_path)
        #TODO add more validity checks
        if "input_output.json" in files or "sols.json" in files:
            valid_probs.append(prob_path)
    valid_probs = sorted(valid_probs)
    #with open(os.path.join(args.source,"valid_problems.json"), "w") as f:
    #    json.dump(valid_probs, f)
    return valid_probs


def get_question(problem_list, prob_index):
    root = problem_list[prob_index]
    #print("get q", root)
    if os.path.exists(os.path.join(root, "question.txt")):
        with open(os.path.join(root, "question.txt")) as f:
            question = f.readlines()
    else:
        print("question prompt not found")
        question = ""
    question = "".join(question)
    return question


def get_solutions(problem_list, prob_index):
    root = problem_list[prob_index]
    if os.path.exists(os.path.join(root, "solutions.json")):
        with open(os.path.join(root, "solutions.json")) as f:
            sols = json.load(f)
    return sols


# ======================= ENTRY POINT USED BY OUR WORKER =======================
# One call evaluates ONE candidate solution against ALL supplied input/output pairs.
# The misleading argument "test" is candidate Python SOURCE, not a unit-test script.
# problem requires input_output: paired inputs/outputs lists and nullable fn_name.
# problem_list/prob_index are legacy arguments; our worker leaves both unset.
def run_test(problem=None, problem_list:List[str]=None, prob_index:int=None, 
        test:str=None, debug:bool=False):
    """
    if test is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """

    if debug:
        print(f"start = {datetime.now().time()}")

    if problem_list is not None:
        root = problem_list[prob_index]


    # Expected outputs already exist in the dataset; this evaluator does not generate tests.
    in_outs = problem["input_output"]
    if debug:
        print(f"test cases json = {in_outs['inputs']} {in_outs['outputs']}")
    
    # A function name means call-based evaluation; None/missing means stdin/stdout.
    if in_outs.get("fn_name") is None:
        which_type = CODE_TYPE.standard_input  # Standard input
        method_name = None
    else:
        which_type = CODE_TYPE.call_based  # Call-based
        method_name = in_outs["fn_name"]
    if debug:
        print(f"loaded json = {datetime.now().time()}")
 
    # Inspection-only shortcut: unused by our worker, which always submits source.
    if test is None:
        return in_outs
    elif test is not None:
        # Disable functionalities that can make destructive changes to the test.
        # Mutates process-wide functions; our worker runs once in a disposable sandbox.
        reliability_guard()
        
        results = []
        # Executable scaffolding: these common imports can satisfy missing candidate imports.
        sol = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"
        if debug:
            print(f"loading test code = {datetime.now().time()}")
 
        # ------------------- A. LOAD A CALL-BASED SOLUTION --------------------
        if which_type == CODE_TYPE.call_based:
            # Append candidate source to the common imports; no test assertions are constructed.
            sol += test
            if debug: # or True:
                print(f"sol = {sol}")
            signal.alarm(timeout)
            try:
                # Execute the constructed source once to obtain a module containing the candidate.
                tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                # Upstream detects this class by a literal substring, not syntax-tree inspection.
                if "class Solution" not in test:
                    tmp = tmp_sol
                else:
                    tmp = tmp_sol.Solution()
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                print(f"type 0 compilation error = {e}")
                results.append(-2)
                return results
            signal.alarm(0)

        # ------------------- B. LOAD A STDIN/STDOUT SOLUTION ------------------
        elif which_type == CODE_TYPE.standard_input:
            # sol
            # Textually wrap the script in code(); this is not a fresh script process per case.
            # Imports before the first body line stay outside; later imports enter the wrapper.
            tmp_test = test.split("\n")

            new_test = []
            for x in tmp_test:
                if (not x.startswith("from ")) and (not x.startswith("import ")):
                    new_test.append("\t" + x + "\n")
                else:
                    new_test.append(x + "\n")
            tmp_test = new_test
            
            new_test = ""
            started = False
            for i in tmp_test:
                if i.startswith("\t") and not started:
                    new_test += "stdin = sys.stdin\nstdout = sys.stdout\n"
                    new_test += "def code():\n"
                    new_test += i
                    started = True
                elif started and ((i.startswith("from ")) or (i.startswith("import "))): 
                    new_test += "\t" + i
                else:
                    new_test += i
            tmp_test = new_test

            sol += tmp_test
            if debug:
                print(f"sol = {sol}")
                # print(f"{o}") 
            method_name = "code"
            signal.alarm(timeout)
            try:
                # Execute the constructed source once to obtain a module containing the candidate.
                tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                tmp = tmp_sol
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                print(f"type 1 compilation error = {e}")
                results.append(-2)
                return results
            signal.alarm(0)
        if debug:
            print(f"get method = {datetime.now().time()}")
 
        try:
            # Resolve the named function/method, or the generated code() wrapper for stdio.
            method = getattr(tmp, method_name)  # get_attr second arg must be str
        except:
            signal.alarm(0)
            e = sys.exc_info()
            print(f"unable to get function error = {e}")
            results.append(-2)
            return results

        # =================== APPLY ALL SUPPLIED TEST CASES ====================
        # The module and Solution instance are reused, so state can persist between cases.
        for index, inputs in enumerate(in_outs["inputs"]):
            # Upstream attempts integer-key recovery; these conversions are heuristic.
            # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)
            try:
                if isinstance(inputs[0], dict):
                    inputs = [{int(k): v for k,v in inputs[0].items()}]
            except:
                True
            try:
                if isinstance(in_outs["outputs"][index], dict):
                    in_outs["outputs"][index] = [{int(k): v for k,v in in_outs["outputs"][index].items()}]
            except:
                True
            try:
                if isinstance(in_outs["outputs"][index][0], dict):
                    in_outs["outputs"][index] = [{int(k): v for k,v in in_outs["outputs"][index][0].items()}]
            except:
                True

            if debug:
                print(f"time: {datetime.now().time()} testing index = {index}  inputs = {inputs}, {type(inputs)}. type = {which_type}")
            # ---------------- A. CALL WITH POSITIONAL ARGUMENTS ----------------
            if which_type == CODE_TYPE.call_based:  # Call-based
                signal.alarm(timeout)
                # Requires stderr.fileno(); our driver captures stderr in a real file.
                faulthandler.enable()
                try:
                    # print("------------")
                    # print(inputs)
                    # Each inputs entry is an argument list: [[1, 2]] passes ONE array argument.
                    output = method(*inputs)

                    # ground truth sequences are not tuples
                    if isinstance(output, tuple):
                        output = list(output)
                    
                    # Compare return values, with upstream tuple/list and nested-output fallbacks.
                    tmp_result = output == in_outs["outputs"][index]
                    if isinstance(in_outs["outputs"][index], list) and in_outs["outputs"][index]:
                        tmp_result = tmp_result or (output == in_outs["outputs"][index][0])

                    # ground truth sequences are not tuples
                    try:
                        if isinstance(output[0], tuple):
                            tmp_result = tmp_result or ([list(x) for x in output] == in_outs["outputs"][index][0])
                    except:
                        True
                    results.append(tmp_result)

                    # reset the alarm
                    signal.alarm(0)
                except Exception as e:
                    signal.alarm(0)
                    faulthandler.disable()
                    # Despite this diagnostic label, we are in the call-based branch.
                    print(f"Standard input runtime error or time limit exceeded error = {e}")
                    results.append(-1)
                    continue
                faulthandler.disable()
                signal.alarm(0)
                if debug:
                    print(f"outputs = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
            # ---------------- B. FEED STDIN AND CAPTURE STDOUT -----------------
            elif which_type == CODE_TYPE.standard_input:  # Standard input
                # Requires stderr.fileno(); our driver captures stderr in a real file.
                faulthandler.enable()
                signal.alarm(timeout)
                passed = False

                # Lists of input/output lines become newline-separated text.
                if isinstance(inputs, list):
                    inputs = "\n".join(inputs)
                if isinstance(in_outs['outputs'][index], list):
                    in_outs['outputs'][index] = "\n".join(in_outs['outputs'][index])

                with Capturing() as output:
                    try:
                        # Patch stdin/open and invoke code(); Capturing collects printed output.
                        call_method(method, inputs)
                        # reset the alarm
                        signal.alarm(0)
                        passed = True
                    except Exception as e:
                        # runtime error or took too long
                        signal.alarm(0)
                        # Despite this diagnostic label, we are in the stdio branch.
                        print(f"Call-based runtime error or time limit exceeded error = {repr(e)}{e}")
                        results.append(-1)
                    signal.alarm(0)

                if not passed:
                    if debug:
                        nl = "\n"
                        if not isinstance(inputs, list):
                            print(f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                        else:
                            print(f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                    continue

                if passed and debug:
                    print(f"==> output = {output}, test outputs = {in_outs['outputs'][index]}")

                # Comparison cascade: accept the first match, starting with stripped text.
                if custom_compare_(output, in_outs['outputs'][index]):
                    tmp_result = True
                    results.append(tmp_result)
                    continue

                # Fallbacks retain upstream permissiveness, rather than strict byte equality.
                # ground truth sequences are expressed as lists not tuples
                if isinstance(output, tuple):
                    output = list(output)

                tmp_result = False
                try:
                    tmp_result = (output == [in_outs["outputs"][index]])
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                        if isinstance(output[0], str):
                            tmp_result = tmp_result or ([e.strip() for e in output] == in_outs["outputs"][index])
                except Exception as e:
                    print(f"Failed check1 exception = {e}")
                    pass

                if tmp_result == True:  
                    results.append(tmp_result)
                    continue

                # The following normalizations mutate the expected-output structure in place.
                # try one more time without \n
                if isinstance(in_outs["outputs"][index], list):
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        in_outs["outputs"][index][tmp_index] = i.split("\n")
                        in_outs["outputs"][index][tmp_index] = [x.strip() for x in in_outs["outputs"][index][tmp_index] if x]
                else:
                    in_outs["outputs"][index] = in_outs["outputs"][index].split("\n")
                    in_outs["outputs"][index] = list(filter(len, in_outs["outputs"][index]))
                    in_outs["outputs"][index] = list(map(lambda x:x.strip(), in_outs["outputs"][index]))

                try:
                    tmp_result = (output == [in_outs["outputs"][index]])
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                except Exception as e:
                    print(f"Failed check2 exception = {e}")
                    pass

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try by converting the output into a split up list too
                if isinstance(output, list):
                    output = list(filter(len, output))

                if debug:
                    nl = "\n"
                    if not isinstance(inputs, list):
                        print(f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}") 
                    else:
                        print(f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}") 
                
                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                try:
                    tmp_result = (output == [in_outs["outputs"][index]])
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                except Exception as e:
                    print(f"Failed check3 exception = {e}")
                    pass

                try:
                    # np.allclose below accepts sufficiently close floating-point outputs.
                    output_float = [float(e) for e in output]
                    gt_float = [float(e) for e in in_outs['outputs'][index]]
                    tmp_result = tmp_result or ((len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
                except Exception as e:
                    pass
                try:
                    if isinstance(output[0], list):
                        output_float = [float(e) for e in output[0]]
                        gt_float = [float(e) for e in in_outs['outputs'][index][0]]
                        tmp_result = tmp_result or ((len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
                except Exception as e:
                    pass

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # Set-based fallbacks can discard token order and duplicate counts.
                # try by converting the stuff into split up list
                if isinstance(in_outs["outputs"][index], list):
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        in_outs["outputs"][index][tmp_index] = set(i.split())
                else:
                    in_outs["outputs"][index] = set(in_outs["outputs"][index].split())

                try:
                    tmp_result = (output == in_outs["outputs"][index])
                except Exception as e:
                    print(f"Failed check4 exception = {e}")
                    continue

                if tmp_result == True:
                    results.append(tmp_result)
                    continue 

                # try by converting the output into a split up list too
                if isinstance(output, list):
                    for tmp_index, i in enumerate(output):
                        output[tmp_index] = i.split()
                    output = list(filter(len, output))
                    for tmp_index, i in enumerate(output):
                        output[tmp_index] = set(i)    
                else:
                    output = output.split()
                    output = list(filter(len, output))
                    output = set(output)

                try:
                    tmp_result = (set(frozenset(s) for s in output) == set(frozenset(s) for s in in_outs["outputs"][index]))
                except Exception as e:
                    print(f"Failed check5 exception = {e}")


                # Last fallback also rounds numeric tokens to three decimal places.
                # if they are all numbers, round so that similar numbers are treated as identical
                try:
                    tmp_result = tmp_result or (set(frozenset(round(float(t),3) for t in s) for s in output) ==\
                        set(frozenset(round(float(t),3) for t in s) for s in in_outs["outputs"][index]))
                except Exception as e:
                    print(f"Failed check6 exception = {e}")
                
                if tmp_result == True and debug:
                    print("PASSED")
 
                results.append(tmp_result)
                
                if debug:
                    nl = "\n"
                    if not isinstance(inputs, list):
                        print(f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                    else:
                        print(f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}") 


    # Normal result: one bool/-1 per case; initialization failures can return [-2].
    return results

# ========================== STDIO COMPARISON HELPERS ==========================
def custom_compare_(output, ground_truth):
    
    if isinstance(output, list):
        output_1 = "\n".join(output)
        if stripped_string_compare(output_1, ground_truth):
            return True

    if isinstance(output, list):
        output_2 = [o.lstrip().rstrip() for o in output]
        output_2 = "\n".join(output_2)
        if stripped_string_compare(output_2, ground_truth):
            return True

    return False

def stripped_string_compare(s1, s2):
    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()
    return s1 == s2

# =========================== STDIO INPUT ADAPTER ==============================
# Only the stdio branch uses this adapter; call-based solutions receive arguments.
def call_method(method, inputs):

    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    # Mocked open() returns case input regardless of filename, unlike normal file I/O.
    @patch('builtins.open', mock_open(read_data=inputs))
    @patch('sys.stdin', StringIO(inputs))
    @patch('sys.stdin.readline', lambda *args: next(inputs_line_iterator))
    @patch('sys.stdin.readlines', lambda *args: inputs.split("\n"))
    @patch('sys.stdin.read', lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit as e:
            pass
        finally:
            pass
    return _inner_call_method(method) 

# ========================== PROCESS RELIABILITY GUARD =========================
# Used by BOTH invocation modes; Modal provides the actual execution isolation.
def reliability_guard(maximum_memory_bytes=None):
    """
    source: https://github.com/openai/human-eval
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    # Our runner does not pass this optional bound; Modal limits sandbox memory.
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None

# ====================== LEGACY CLI: UNUSED BY OUR WRAPPER =====================
# Retained verbatim for equivalence, including stale helpers/call signatures.
def main(args):
    print(args)
    problem_list = sorted(get_valid_problems(args.source))
    prob_index = args.number

    # This checks it correctly loaded. remove this later
    assert prob_index < len(problem_list)

    if args.data == "q" or args.data == "question":
        tmp = get_question(problem_list, prob_index)
        print("q", tmp)
    elif args.data in ["solutions", "sol", "s",]:
        tmp = get_solutions(problem_list, prob_index)
        print("sol", tmp)
    elif args.data == "starter":
        tmp = get_starter(problem_list, prob_index)
        print("starter", tmp)
    elif args.data in ["test", "t"]:
        # test it with sols
        sols = get_solutions(problem_list, prob_index)
        tmp = run_test(problem_list, prob_index, test=sols[0])

        print("results = ", tmp)
        print("-2 = compile error, -1 is runtime error, False failed test, True passed test")

if __name__ == "__main__":
    args = parse_args()
    main(args)
