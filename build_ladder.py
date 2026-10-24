#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builder & Verifier for the ML Memory & Cache Systems Ladder:
  1. Generates `/home/droid/ml-memory-ladder/ml_memory_ladder.html`
     (following the exact pattern of `senior-lab/web/ml_ladder.html`, with
      Leitner spaced-repetition Quiz + interactive Python & C++ Code labs +
      deep line-by-line & every-variable inspector).
  2. Generates `/home/droid/ml-memory-ladder/STUDY_GUIDE.md`
     (complete textbook-grade interactive study guide with every line of C++
      and Python code and every variable explained in detail).
"""

import json
import os
import re
from ladder_data_part1 import LEVELS_1_4, QUIZ_1_4, EXERCISES_1_4
from ladder_data_part2 import LEVELS_5_8, QUIZ_5_8, EXERCISES_5_8
from enrich_and_verify import apply_enrichments

LEVELS = LEVELS_1_4 + LEVELS_5_8
QUIZ, EXERCISES = apply_enrichments(QUIZ_1_4 + QUIZ_5_8, EXERCISES_1_4 + EXERCISES_5_8)

GLOSS_CPP = [
    [r"\bcalloc\b", "calloc(count, size)", "Allocates `count * size` bytes on the heap, checks the multiplication for integer overflow, and zeroes every byte. Returns `NULL` (0) if memory is exhausted."],
    [r"\bmalloc\b", "malloc(bytes)", "Allocates `bytes` of uninitialized heap memory. Returns `NULL` on failure; does NOT zero the bytes or check `n * sizeof(T)` for overflow."],
    [r"\bfree\b", "free(ptr)", "Returns the heap block at `ptr` back to the allocator. Does NOT change the variable `ptr` itself (leaving `ptr` as a dangling pointer unless set to `NULL`). `free(NULL)` is a safe no-op."],
    [r"\bNULL\b", "NULL", "Address `0` — the special pointer value meaning 'points to no memory block'. Dereferencing `NULL` (`*p` or `p[i]`) immediately faults (`SIGSEGV`)."],
    [r"\bconst\b", "const T*", "A pointer to read-only elements: the function promises not to modify the array through this pointer."],
    [r"\blong long\b", "long long", "A 64-bit signed integer in C++ (at least 64 bits, up to ~9.22e18), used to prevent 32-bit `int` multiplication overflow."],
    [r"->", "ptr->field", "Dereferences a struct pointer `ptr` and accesses `field` (shorthand for `(*ptr).field`). Unsafe if `ptr` is `NULL` or already freed!"],
    [r"\*\w+\s*=", "*out = val", "Writes `val` into the memory slot pointed to by pointer `out` (pointer dereference on the left side of `=`)."],
    [r"&\w+", "&var", "The 'address-of' operator: returns a pointer holding the memory address where `var` lives."],
    [r"\?", "cond ? a : b", "C++ ternary expression: evaluates to `a` when `cond` is non-zero (true), otherwise `b`."],
    [r"\|\|", "||", "Logical OR: true if either the left side or the right side is true. Short-circuits if the left side is true."],
    [r"&&", "&&", "Logical AND: true only when both sides are true. Short-circuits if the left side is false."],
    [r"%", "a % b", "Modulo (remainder after integer division): `count % window_size` wraps an index around `0 .. window_size - 1` for circular ring buffers."],
]

GLOSS_PY = [
    [r"\bNone\b", "None", "Python's null value meaning 'no object here' (used to model C's `NULL` pointer return on allocation failure). Check with `is None`."],
    [r"\bis None\b", "x is None", "Tests whether `x` is the singleton `None` object."],
    [r"//", "a // b", "Floor (integer) division in Python: `uint32_max // elem_bytes` divides and rounds down to a whole integer without floating-point rounding."],
    [r"&", "a & b", "Bitwise AND: `(n_elems * elem_bytes) & 0xFFFFFFFF` keeps only the lowest 32 bits, modeling 32-bit unsigned integer wraparound."],
    [r"\.append\(", "list.append(x)", "Adds `x` to the end of a list in-place, growing the list by 1 slot."],
    [r"\.pop\(", "list.pop()", "Removes and returns the last item of a list in-place."],
    [r"\.update\(", "dict.update(other)", "Copies all key/value pairs from `other` into `dict` in-place (modeling `module.load_state_dict(checkpoint)`)."],
    [r"\bisinstance\(", "isinstance(x, dict)", "Returns `True` if `x` is a dictionary (used here to check whether an entry in `history` is still a live autograd graph node dict vs. a detached float)."],
    [r"\bmin\(", "min(a, b)", "Returns the smaller value (or smallest item in a list, such as the best-fit free block in `CUDACachingAllocator`)."],
    [r"\bmax\(", "max(a, b)", "Returns the larger value (such as `max(cpu_free_tick, stream2_done_tick)` when synchronizing across CUDA streams)."],
]


def transpile_cpp_to_js(cpp_code: str, fn_name: str, ex_id: str) -> str:
    """
    Converts a C++ exercise function (and its helpers) into executable ES5 JS
    inside the browser/Node worker so C++ exercises can be run and graded
    against the exact same test suite!
    """
    src = cpp_code
    # Strip single-line comments
    src = re.sub(r"//[^\n]*", "", src)
    # Replace C++ types in parameter lists and variable declarations
    # Handle function signatures: (void|int|long long|double|int\*) name(...)
    def repl_fn(m):
        name = m.group(2)
        params = m.group(3)
        p_names = []
        for p in params.split(","):
            p = p.strip()
            if not p:
                continue
            toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", p)
            if toks:
                p_names.append(toks[-1])
        return "function " + name + "(" + ", ".join(p_names) + ") {"

    src = re.sub(
        r"\b(void|int\s*\*|int|long\s+long|double)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*\{",
        repl_fn,
        src,
    )
    # Fixed-size C stack arrays: `int old_buf[64];` or `int keys[32], q_keys[32];`
    src = re.sub(
        r"\bint\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*\d+\s*\]\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*\d+\s*\]\s*;",
        r"var \1 = [], \2 = [];",
        src,
    )
    src = re.sub(
        r"\bint\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*\d+\s*\]\s*=\s*\{\s*([^}]+)\s*\}\s*;",
        r"var \1 = [\2];",
        src,
    )
    src = re.sub(
        r"\bint\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*\d+\s*\]\s*;",
        r"var \1 = [];",
        src,
    )
    # C casts `(long long)`
    src = re.sub(r"\(\s*long\s+long\s*\)", "", src)
    # Variable declarations: `long long x = ...`, `int* x = ...`, `int x = ...`, `double x = ...`
    src = re.sub(r"\b(long\s+long|int\s*\*|int|double)\s+", "var ", src)
    # Pointer dereferences `*step_count += 1;` -> `step_count[0] += 1;`
    src = re.sub(r"\(\*([A-Za-z_][A-Za-z0-9_]*)\)", r"\1[0]", src)
    src = re.sub(r"\*([A-Za-z_][A-Za-z0-9_]*)", r"\1[0]", src)
    # Address-of `&p1` -> `_box_p1`? Wait: for exercises using `&p1` or `&count1`,
    # let's wrap scalar locals that have `&var` taken into 1-element arrays `[val]`!
    addr_vars = set(re.findall(r"&([A-Za-z_][A-Za-z0-9_]*)", src))
    for v in addr_vars:
        # Replace `var v = expr;` with `var v = [expr];`
        src = re.sub(
            rf"\bvar\s+{v}\s*=\s*([^;]+);",
            rf"var {v} = [\1];",
            src,
        )
        # Replace `&v` with `v`
        src = re.sub(rf"&{v}\b", v, src)
        # Replace bare uses of `v` (not `v[0]`) after declaration with `v[0]`
        # Specifically in assignments like `var p2 = p1;` or `*out_p = p2;`
        src = re.sub(rf"\b{v}\b(?!\s*\[)(?!\s*=)", f"{v}[0]", src)
        # Fix `var [v[0]]` back to `var v`
        src = src.replace(f"var {v}[0]", f"var {v}")
        # Fix function call argument `step_ada(p1[0], ...)` -> `step_ada(p1, ...)`
        src = re.sub(rf"(step_ada|write_ring)\(([^)]*)\b{v}\[0\]", rf"\1(\2{v}", src)
        src = re.sub(rf"(step_ada|write_ring)\(([^)]*)\b{v}\[0\]", rf"\1(\2{v}", src)

    # NULL -> null
    src = re.sub(r"\bNULL\b", "null", src)
    return src


def build_html() -> str:
    """Builds the standalone interactive `ml_memory_ladder.html` file."""
    ladder_json = json.dumps({"LEVELS": LEVELS, "BANK": QUIZ}, ensure_ascii=False)
    exercises_json = json.dumps(EXERCISES, ensure_ascii=False)
    gloss_cpp_json = json.dumps(GLOSS_CPP, ensure_ascii=False)
    gloss_py_json = json.dumps(GLOSS_PY, ensure_ascii=False)

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>ML Memory &amp; Cache Systems Ladder (C++ &amp; Python)</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0f1419;color:#e6e6e6;font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;-webkit-text-size-adjust:100%}}
#app{{max-width:780px;margin:0 auto;padding:14px 16px 48px}}
h1{{font-size:21px;margin:6px 0 2px}}
.sub{{color:#8fa1b3;font-size:14px;margin-bottom:10px}}
.bar{{height:10px;background:#1f2933;border-radius:5px;overflow:hidden;margin:4px 0 2px}}
.bar i{{display:block;height:100%;background:#4cc38a;width:0;transition:width .25s}}
.meta{{display:flex;justify-content:space-between;color:#8fa1b3;font-size:13px;flex-wrap:wrap;gap:6px}}
.card{{background:#18202a;border:1px solid #263445;border-radius:12px;padding:14px;margin:12px 0}}
.q{{font-size:18px;font-weight:600;margin:0 0 12px}}
button{{font:inherit;color:#e6e6e6;background:#243242;border:1px solid #33475c;border-radius:10px;padding:11px 12px;margin:6px 0;width:100%;text-align:left;cursor:pointer}}
button:active{{background:#2e4258}}
button.idk{{background:#2a2433;border-color:#4a3d5c;color:#c9b8e0}}
button.ok{{background:#17402c;border-color:#4cc38a}}
button.bad{{background:#47202a;border-color:#e05d6f}}
button.go{{background:#2f6fed;border-color:#2f6fed;text-align:center;font-weight:600}}
button[disabled]{{opacity:.85;cursor:default}}
.why{{margin-top:10px;padding:10px 12px;border-left:4px solid #4cc38a;background:#13261e;border-radius:6px;font-size:14.5px}}
.why.bad{{border-color:#e05d6f;background:#2a161b}}
.why.idk{{border-color:#a98bd6;background:#221b2e}}
.why.info{{border-color:#6aa9ff;background:#131f30}}
.tag{{display:inline-block;font-size:12px;padding:2px 8px;border-radius:9px;background:#243242;color:#8fa1b3;margin-right:6px;margin-bottom:4px}}
.tag.bug{{background:#3b232e;color:#ff9eb0;border:1px solid #6e354b}}
.small{{font-size:13px;color:#8fa1b3}}
select{{font:inherit;background:#243242;color:#e6e6e6;border:1px solid #33475c;border-radius:8px;padding:6px}}
a,.lnk{{color:#6aa9ff;cursor:pointer;text-decoration:underline}}
.tabs{{display:flex;gap:8px;margin:8px 0}}
.tabs button{{margin:0;text-align:center;padding:9px;font-weight:600}}
.tabs button.on{{background:#2f6fed;border-color:#2f6fed}}
textarea{{width:100%;min-height:240px;font:13.5px/1.45 ui-monospace,Menlo,Consolas,monospace;background:#0b0f14;color:#d8e6f5;border:1px solid #33475c;border-radius:8px;padding:10px;white-space:pre;overflow:auto;-webkit-appearance:none;resize:vertical}}
.sym{{display:flex;flex-wrap:wrap;gap:5px;margin:6px 0}}
.sym button{{width:auto;margin:0;padding:7px 10px;text-align:center;font:14px ui-monospace,Menlo,Consolas,monospace}}
.row{{display:flex;gap:8px}}.row button{{text-align:center}}
pre.sol{{white-space:pre-wrap;background:#0b0f14;border:1px solid #2b3b4e;border-radius:8px;padding:10px;font:13px/1.45 ui-monospace,Menlo,Consolas,monospace;overflow:auto;margin:6px 0}}
.res{{font:13px/1.45 ui-monospace,Menlo,Consolas,monospace;margin:4px 0;word-break:break-word}}
.res.p{{color:#4cc38a}}.res.f{{color:#ff8c9a}}
button.lock{{opacity:.5}}
table.vtable{{width:100%;border-collapse:collapse;margin-top:6px;font-size:13.5px}}
table.vtable th,table.vtable td{{border-bottom:1px solid #263849;padding:7px 8px;text-align:left;vertical-align:top}}
table.vtable th{{color:#8fa1b3;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
table.vtable td.vname{{font-family:ui-monospace,Menlo,Consolas,monospace;color:#7bc98c;white-space:nowrap;font-weight:600}}
table.vtable td.lcode{{font-family:ui-monospace,Menlo,Consolas,monospace;color:#ffd479;background:#0b0f14;white-space:pre-wrap;width:44%}}
details summary{{cursor:pointer;user-select:none}}
</style></head><body><div id="app"></div>
<script>
/* ML Memory & Cache Systems Ladder data bank: 8 levels x 8 questions (64 questions) + 16 C++/Python exercises.
   Follows the exact pattern of ml_ladder.html. ES5 compatible. */
window.MLLADDER = {ladder_json};
window.MLEX = {exercises_json};
window.MLGLOSS = {{ cpp: {gloss_cpp_json}, py: {gloss_py_json} }};
</script>
<script>
(function(){{
'use strict';
var L = window.MLLADDER.LEVELS, NLEV = L.length, BANK = window.MLLADDER.BANK, EX = window.MLEX, PER = 8, KEY = 'mlmemladder.v1';
var app = document.getElementById('app');
var S = {{level:1, top:1, box:{{}}, due:[], n:0, stats:{{ans:0, right:0, idk:0}}, done:false, code:{{}}, drafts:{{}}, mode:'quiz', view:'list', lang:'py'}};
var cur = null, answered = false;

function load(){{try{{var r=localStorage.getItem(KEY);if(r){{var o=JSON.parse(r);if(o&&o.level)S=o;}}}}catch(e){{}}}}
function save(){{try{{localStorage.setItem(KEY,JSON.stringify(S));}}catch(e){{}}}}
function rnd(n){{return Math.floor(Math.random()*n);}}
function shuffle(a){{a=a.slice();for(var i=a.length-1;i>0;i--){{var j=rnd(i+1),t=a[i];a[i]=a[j];a[j]=t;}}return a;}}
function esc(s){{return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}}
function idsAt(l){{var r=[];for(var i=0;i<BANK.length;i++)if(BANK[i].l===l)r.push(i);return r;}}
function bx(i){{return S.box[i]===undefined?-1:S.box[i];}}
function mastered(l){{var c=0,ids=idsAt(l);for(var k=0;k<ids.length;k++)if(bx(ids[k])>=2)c++;return c;}}

function pick(){{
  var k;
  for(k=0;k<S.due.length;k++){{if(S.due[k].at<=S.n){{var d=S.due.splice(k,1)[0];return d.id;}}}}
  var ids=idsAt(S.level),fresh=[];
  for(k=0;k<ids.length;k++)if(bx(ids[k])<0)fresh.push(ids[k]);
  if(fresh.length&&Math.random()<0.85)return fresh[rnd(fresh.length)];
  var pool=[];
  for(k=0;k<BANK.length;k++){{if(BANK[k].l<=S.level&&bx(k)>=0&&bx(k)<9)pool.push(k);}}
  var weak=[];for(k=0;k<ids.length;k++)if(bx(ids[k])<2&&bx(ids[k])>=0)weak.push(ids[k]);
  if(weak.length&&Math.random()<0.6)return weak[rnd(weak.length)];
  if(pool.length)return pool[rnd(pool.length)];
  if(fresh.length)return fresh[rnd(fresh.length)];
  return ids[rnd(ids.length)];
}}

function tabsHtml(){{
  var solved=0;for(var i=0;i<EX.length;i++)if(S.code[EX[i].id]&&S.code[EX[i].id].ok)solved++;
  return '<div class="tabs">' +
    '<button id="tq" class="'+(S.mode==='quiz'?'on':'')+'">Quiz (64 Qs)</button>' +
    '<button id="tc" class="'+(S.mode==='code'?'on':'')+'">Code Labs ('+solved+'/'+EX.length+')</button>' +
    '<button id="tg" class="'+(S.mode==='guide'?'on':'')+'">Full Study Guide</button>' +
  '</div>';
}}

function wireTabs(){{
  var q=document.getElementById('tq'),c=document.getElementById('tc'),g=document.getElementById('tg');
  if(q)q.onclick=function(){{S.mode='quiz';S.view='list';save();render();}};
  if(c)c.onclick=function(){{S.mode='code';S.view='list';save();render();}};
  if(g)g.onclick=function(){{S.mode='guide';save();render();}};
}}

function header(){{
  var m=mastered(S.level),pct=Math.round(100*m/PER);
  return '<h1>ML Memory &amp; Cache Systems Ladder</h1>' +
    '<div class="sub">Low-level C++, Python, GPU Caching Allocator &amp; KV-Cache Bug Fixes &mdash; built from your real OSS bug ledger.</div>' +
    '<div class="meta"><span><b>Level '+S.level+' of '+NLEV+':</b> '+esc(L[S.level-1])+'</span><span>'+m+'/'+PER+' mastered</span></div>' +
    '<div class="bar"><i style="width:'+pct+'%"></i></div>' +
    '<div class="meta"><span>Answered '+S.stats.ans+' &middot; correct '+S.stats.right+' &middot; unsure '+S.stats.idk+'</span>' +
    '<span class="lnk" id="menu">jump to any level</span></div>' + tabsHtml();
}}

function exSolved(id){{return !!(S.code[id]&&S.code[id].ok);}}
function levelDone(l){{for(var i=0;i<EX.length;i++)if(EX[i].l===l&&!exSolved(EX[i].id))return false;return true;}}
function unlocked(l){{return true;}} /* All 8 levels unlocked so you can study or jump anywhere freely */
function exById(id){{for(var i=0;i<EX.length;i++)if(EX[i].id===id)return EX[i];return null;}}
function nextEx(){{for(var i=0;i<EX.length;i++)if(!exSolved(EX[i].id))return EX[i];return null;}}

function langBar(){{
  return '<div class="tabs">' +
    '<button id="lp" class="'+(S.lang==='py'?'on':'')+'">Python (Pyodide / In-Page)</button>' +
    '<button id="lc" class="'+(S.lang==='cpp'?'on':'')+'">C++ (Low-Level Pointer &amp; Heap Sim)</button>' +
  '</div>';
}}

function wireLang(rerender){{
  var p=document.getElementById('lp'),c=document.getElementById('lc');
  if(p)p.onclick=function(){{S.lang='py';save();rerender();}};
  if(c)c.onclick=function(){{S.lang='cpp';save();rerender();}};
}}

function explainHtml(ex, openByDefault){{
  var isPy = (S.lang === 'py');
  var lines = isPy ? ex.lines_py : ex.lines_cpp;
  var sol = isPy ? ex.py_sol : ex.cpp_sol;
  var gList = isPy ? window.MLGLOSS.py : window.MLGLOSS.cpp;
  var op = openByDefault ? ' open' : '';
  var h = '<div class="why info" style="margin-top:10px"><b>Real Bug Connection:</b> <code>' + esc(ex.real_bug) + '</code></div>';

  /* 1. Every Variable & Parameter Table */
  h += '<details' + op + ' class="why" style="margin-top:8px"><summary><b>1. What Every Variable &amp; Parameter Means (' + ex.v.length + ' variables)</b></summary>';
  h += '<table class="vtable"><thead><tr><th>Variable / Parameter</th><th>Exact Role in RAM / VRAM &amp; What Goes Wrong</th></tr></thead><tbody>';
  for(var i=0;i<ex.v.length;i++){{
    h += '<tr><td class="vname">' + esc(ex.v[i][0]) + '</td><td>' + esc(ex.v[i][1]) + '</td></tr>';
  }}
  h += '</tbody></table></details>';

  /* 2. Line-by-Line Code Walkthrough */
  h += '<details' + op + ' class="why" style="margin-top:8px"><summary><b>2. Line-by-Line Code Walkthrough (' + (isPy ? 'Python' : 'C++') + ')</b></summary>';
  h += '<table class="vtable"><thead><tr><th>Code Line</th><th>What This Line Does Step-by-Step</th></tr></thead><tbody>';
  for(var j=0;j<lines.length;j++){{
    h += '<tr><td class="lcode">' + esc(lines[j][0]) + '</td><td>' + esc(lines[j][1]) + '</td></tr>';
  }}
  h += '</tbody></table></details>';

  /* 3. Symbol & Keyword Glossary */
  var matched = [];
  for(var m=0;m<gList.length;m++){{
    var rx = new RegExp(gList[m][0]);
    if(rx.test(sol)) matched.push(gList[m]);
  }}
  if(matched.length){{
    h += '<details class="why" style="margin-top:8px"><summary><b>3. Symbols &amp; Keywords Used Here (' + (isPy ? 'Python' : 'C++') + ')</b></summary><div style="margin-top:6px">';
    for(var k=0;k<matched.length;k++){{
      h += '<div style="margin-bottom:6px"><code><b>' + esc(matched[k][1]) + '</b></code> &mdash; ' + esc(matched[k][2]) + '</div>';
    }}
    h += '</div></details>';
  }}
  return h;
}}

/* ---------------- In-Page C++ & Python Test Runners ---------------- */
var PYODIDE='https://cdn.jsdelivr.net/pyodide/v0.24.1/full/',PYW=null,PYREADY=false,PYERR='',PYQ=[],PYBUSY=null;

function ensurePy(){{
  if(PYW)return;
  try{{
    var wsrc = "importScripts('" + PYODIDE + "pyodide.js');" +
      "var P=null,out='';" +
      "var ready=loadPyodide({{indexURL:'" + PYODIDE + "'}}).then(function(p){{P=p;postMessage({{ready:true}});}},function(e){{postMessage({{loadErr:String(e)}});}});" +
      "onmessage=function(e){{var d=e.data;ready.then(function(){{try{{var r=P.runPython(d.script);postMessage({{res:r}});}}catch(err){{postMessage({{err:String(err)}});}}}});}};";
    PYW = new Worker(URL.createObjectURL(new Blob([wsrc],{{type:'application/javascript'}})));
    PYW.onmessage = function(e){{
      var d = e.data;
      if(d.ready){{PYREADY=true;pyNext();return;}}
      if(d.loadErr){{PYERR=d.loadErr;pyFlush();return;}}
      var job=PYBUSY;PYBUSY=null;
      if(job){{clearTimeout(job.t);job.cb(d);}}
      pyNext();
    }};
    PYW.onerror = function(e){{PYERR=String(e&&e.message||'worker error');pyFlush();}};
  }}catch(e){{PYERR=String(e);}}
}}

function pyFlush(){{
  var q=PYQ;PYQ=[];
  if(PYBUSY){{q.unshift(PYBUSY);clearTimeout(PYBUSY.t);PYBUSY=null;}}
  for(var i=0;i<q.length;i++) q[i].cb({{fallback:true}});
}}

function pyNext(){{
  if(!PYREADY||PYBUSY||!PYQ.length)return;
  var job=PYBUSY=PYQ.shift();
  job.t=setTimeout(function(){{if(PYBUSY!==job)return;PYBUSY=null;try{{PYW.terminate();}}catch(e){{}}PYW=null;PYREADY=false;job.cb({{err:'Timed out after 8 seconds (infinite loop?).'}});}},8000);
  PYW.postMessage({{script:job.script}});
}}

/* Fast offline Python-subset & C++ transpiler so tests also run instantaneously even without internet! */
function transpilePyToJs(pyCode, fnName){{
  var lines = pyCode.replace(/\\r\\n/g, '\\n').split('\\n');
  var out = [];
  var stack = [0];
  for(var i=0;i<lines.length;i++){{
    var raw = lines[i];
    var stripped = raw.replace(/#.*$/, '');
    if(!stripped.trim()) continue;
    var indent = stripped.match(/^\\s*/)[0].length;
    var s = stripped.trim();
    while(stack.length > 1 && indent < stack[stack.length-1]){{
      stack.pop();
      out.push('}}');
    }}
    if(/^def\\s+/.test(s)){{
      var m = s.match(/^def\\s+([A-Za-z0-9_]+)\\s*\\(([^)]*)\\)\\s*:/);
      if(!m) throw new Error('Syntax error on line '+(i+1)+': '+s);
      out.push('function '+m[1]+'('+m[2]+') {{');
      stack.push(indent + 4);
      continue;
    }}
    if(/^(if|elif|while|for|else)\\b/.test(s)){{
      var mIf = s.match(/^if\\s+(.*):$/);
      if(mIf){{
        out.push('if (' + pyExprToJs(mIf[1]) + ') {{');
        stack.push(indent + 4);
        continue;
      }}
      var mElif = s.match(/^elif\\s+(.*):$/);
      if(mElif){{
        out.push('else if (' + pyExprToJs(mElif[1]) + ') {{');
        stack.push(indent + 4);
        continue;
      }}
      if(/^else\\s*:$/.test(s)){{
        out.push('else {{');
        stack.push(indent + 4);
        continue;
      }}
      var mWh = s.match(/^while\\s+(.*):$/);
      if(mWh){{
        out.push('while (' + pyExprToJs(mWh[1]) + ') {{');
        stack.push(indent + 4);
        continue;
      }}
      var fm = s.match(/^for\\s+([A-Za-z0-9_]+)\\s+in\\s+range\\((.+)\\)\\s*:$/);
      if(fm){{
        var v = fm[1], rarg = pyExprToJs(fm[2]);
        out.push('for (var '+v+' = 0; '+v+' < ('+rarg+'); '+v+'++) {{');
        stack.push(indent + 4);
        continue;
      }}
      var fm2 = s.match(/^for\\s+([A-Za-z0-9_]+)\\s+in\\s+(.*):$/);
      if(fm2){{
        var v2 = fm2[1], coll = pyExprToJs(fm2[2]), idxv = '_i_'+i;
        out.push('var _c_'+i+' = '+coll+'; for (var '+idxv+' = 0; '+idxv+' < _c_'+i+'.length; '+idxv+'++) {{ var '+v2+' = _c_'+i+'['+idxv+'];');
        stack.push(indent + 4);
        continue;
      }}
    }}
    if(/^return\\b/.test(s)){{
      var rexpr = s.replace(/^return\\s*/, '');
      out.push('return ' + pyExprToJs(rexpr) + ';');
      continue;
    }}
    /* slice assignment or regular statement */
    var assignM = s.match(/^([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*(.*)$/);
    if(assignM){{
      out.push('var ' + assignM[1] + ' = ' + pyExprToJs(assignM[2]) + ';');
    }} else {{
      out.push(pyExprToJs(s) + ';');
    }}
  }}
  while(stack.length > 1){{ stack.pop(); out.push('}}'); }}
  return out.join('\\n');
}}

function replaceCall(str, fnName, replacer){{
  var rx = new RegExp('\\\\b' + fnName + '\\\\(');
  var guard = 0;
  while(guard++ < 40){{
    var m = rx.exec(str);
    if(!m) break;
    var start = m.index;
    var openIdx = start + m[0].length - 1;
    var depth = 1, i = openIdx + 1;
    for(; i < str.length && depth > 0; i++){{
      if(str[i] === '(') depth++;
      else if(str[i] === ')') depth--;
    }}
    if(depth !== 0) break;
    var inner = str.slice(openIdx + 1, i - 1);
    var args = [], cur = '', d = 0;
    for(var j = 0; j < inner.length; j++){{
      var ch = inner[j];
      if(ch === '(' || ch === '[' || ch === '{{') d++;
      else if(ch === ')' || ch === ']' || ch === '}}') d--;
      if(ch === ',' && d === 0){{ args.push(cur.trim()); cur = ''; }}
      else cur += ch;
    }}
    if(cur.trim() || args.length > 0) args.push(cur.trim());
    var rep = replacer(args, inner);
    str = str.slice(0, start) + rep + str.slice(i);
  }}
  return str;
}}

function pyExprToJs(e){{
  e = e.trim();
  /* Handle `[0] * count` or `storage + [0] * (n)` */
  e = e.replace(/\\[0\\]\\s*\\*\\s*\\(([^)]+)\\)/g, '_zeros($1)');
  e = e.replace(/\\[0\\]\\s*\\*\\s*([A-Za-z0-9_]+)/g, '_zeros($1)');
  /* Handle list comprehensions: `[expr for v in range(n)]` or `[b for b in pool if b >= req_bytes]` */
  var lcRange = e.match(/^\\[\\s*(.+?)\\s+for\\s+([A-Za-z0-9_]+)\\s+in\\s+range\\((.+)\\)\\s*\\]$/);
  if(lcRange){{
    return '_lcRange(' + pyExprToJs(lcRange[3]) + ', function(' + lcRange[2] + '){{ return ' + pyExprToJs(lcRange[1]) + '; }})';
  }}
  var lcFilt = e.match(/^\\[\\s*([A-Za-z0-9_]+)\\s+for\\s+([A-Za-z0-9_]+)\\s+in\\s+([A-Za-z0-9_]+)\\s+if\\s+(.+?)\\s*\\]$/);
  if(lcFilt){{
    return lcFilt[3] + '.filter(function(' + lcFilt[2] + '){{ return ' + pyExprToJs(lcFilt[4]) + '; }})';
  }}
  /* Handle `sum(x['val'] if isinstance(x, dict) else x for x in history)` */
  if(/sum\\(x\\['val'\\]\\s+if\\s+isinstance\\(x,\\s*dict\\)\\s+else\\s+x\\s+for\\s+x\\s+in\\s+history\\)/.test(e)){{
    e = e.replace(/sum\\(x\\['val'\\]\\s+if\\s+isinstance\\(x,\\s*dict\\)\\s+else\\s+x\\s+for\\s+x\\s+in\\s+history\\)/g,
      '_sumHistVal(history)');
  }}
  if(/sum\\(x\\['act_bytes'\\]\\s+for\\s+x\\s+in\\s+history\\s+if\\s+isinstance\\(x,\\s*dict\\)\\)/.test(e)){{
    e = e.replace(/sum\\(x\\['act_bytes'\\]\\s+for\\s+x\\s+in\\s+history\\s+if\\s+isinstance\\(x,\\s*dict\\)\\)/g,
      '_sumHistAct(history)');
  }}
  /* Handle slice `a[b : c]` or `a[:c]` */
  e = e.replace(/([A-Za-z0-9_\\]\\['"]+)\\[\\s*:\\s*([^\\]]+)\\]/g, '$1.slice(0, $2)');
  e = e.replace(/([A-Za-z0-9_\\]\\['"]+)\\[([^:\\]]+)\\s*:\\s*([^\\]]+)\\]/g, '$1.slice($2, $3)');
  /* Handle ternary `A if COND else B` */
  var tern = e.match(/^(.+?)\\s+if\\s+(.+?)\\s+else\\s+(.+)$/);
  if(tern){{
    return '((' + pyExprToJs(tern[2]) + ') ? (' + pyExprToJs(tern[1]) + ') : (' + pyExprToJs(tern[3]) + '))';
  }}
  /* Floor division `a // b` */
  e = e.replace(/(\\([^)]+\\)|[A-Za-z0-9_]+)\\s*\\/\\/\\s*([A-Za-z0-9_]+)/g, 'Math.floor($1 / $2)');
  e = e.replace(/\\bis\\s+None\\b/g, '=== null');
  e = e.replace(/\\bis\\s+not\\s+None\\b/g, '!== null');
  e = e.replace(/\\bNone\\b/g, 'null');
  e = e.replace(/\\bTrue\\b/g, 'true');
  e = e.replace(/\\bFalse\\b/g, 'false');
  e = e.replace(/\\band\\b/g, '&&');
  e = e.replace(/\\bor\\b/g, '||');
  e = e.replace(/\\bnot\\s+/g, '!');
  e = replaceCall(e, 'len', function(a, inner){{ return '(' + inner + ').length'; }});
  e = replaceCall(e, 'list', function(a, inner){{ return '(' + inner + ').slice()'; }});
  e = replaceCall(e, 'sum', function(a, inner){{ return '_sum(' + inner + ')'; }});
  e = replaceCall(e, 'max', function(a, inner){{ return '_max(' + inner + ')'; }});
  e = replaceCall(e, 'min', function(a, inner){{ return '_min(' + inner + ')'; }});
  e = replaceCall(e, 'round', function(a, inner){{ return '_round(' + a[0] + ', ' + a[1] + ')'; }});
  e = e.replace(/\\.append\\(/g, '.push(');
  e = e.replace(/\\.remove\\(/g, '._removeVal(');
  e = e.replace(/\\.update\\(/g, '._updateDict(');
  /* List concatenation `base + list(prompt2_keys)` or `storage + _zeros(...)` */
  e = e.replace(/\\b(storage|base)\\s*\\+\\s*/g, '$1.concat(');
  if(/\\b(storage|base)\\.concat\\(/.test(e)) e = e + ')';
  return e;
}}

var JS_HELPERS =
  "Array.prototype._removeVal = function(v){{ var i = this.indexOf(v); if(i>=0) this.splice(i,1); }};\\n" +
  "Object.prototype._updateDict = function(o){{ if(o) for(var k in o) if(Object.prototype.hasOwnProperty.call(o,k)) this[k] = o[k]; }};\\n" +
  "function _zeros(n){{ var a=[]; for(var i=0;i<n;i++) a.push(0); return a; }}\\n" +
  "function _sum(a){{ var s=0; for(var i=0;i<a.length;i++) s+=a[i]; return s; }}\\n" +
  "function _max(a,b){{ if(arguments.length>1) return a>b?a:b; var m=a[0]; for(var i=1;i<a.length;i++) if(a[i]>m) m=a[i]; return m; }}\\n" +
  "function _min(a,b){{ if(arguments.length>1) return a<b?a:b; var m=a[0]; for(var i=1;i<a.length;i++) if(a[i]<m) m=a[i]; return m; }}\\n" +
  "function _round(v,d){{ var p=Math.pow(10,d); return Math.round(v*p)/p; }}\\n" +
  "function _lcRange(n,fn){{ var a=[]; for(var i=0;i<n;i++) a.push(fn(i)); return a; }}\\n" +
  "function _sumHistVal(h){{ var s=0; for(var i=0;i<h.length;i++){{ var x=h[i]; s += (x && typeof x==='object') ? x.val : x; }} return s; }}\\n" +
  "function _sumHistAct(h){{ var s=0; for(var i=0;i<h.length;i++){{ var x=h[i]; if(x && typeof x==='object') s += x.act_bytes; }} return s; }}\\n";

function runPyFastJs(ex, code){{
  var jsBody = JS_HELPERS + transpilePyToJs(code, ex.py_name) + "\\nreturn " + ex.py_name + ";";
  var fn = (new Function(jsBody))();
  return gradeFn(fn, ex.tests);
}}

function transpileCppToJsBrowser(cppCode){{
  var src = cppCode.replace(/\\/\\/[^\\n]*/g, '');
  src = src.replace(/\\b(void|int\\s*\\*|int|long\\s+long|double)\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\(([^)]*)\\)\\s*\\{{/g, function(_, ret, name, params){{
    var pnames = [];
    var ps = params.split(',');
    for(var i=0;i<ps.length;i++){{
      var p = ps[i].trim();
      if(!p) continue;
      var toks = p.match(/[A-Za-z_][A-Za-z0-9_]*/g);
      if(toks) pnames.push(toks[toks.length-1]);
    }}
    return 'function ' + name + '(' + pnames.join(', ') + ') {{';
  }});
  src = src.replace(/\\bint\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\[\\s*\\d+\\s*\\]\\s*,\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*\\[\\s*\\d+\\s*\\]\\s*;/g, 'var $1 = [], $2 = [];');
  src = src.replace(/\\bint\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\[\\s*\\d+\\s*\\]\\s*=\\s*\\{{\\s*([^}}]+)\\s*\\}}\\s*;/g, 'var $1 = [$2];');
  src = src.replace(/\\bint\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\[\\s*\\d+\\s*\\]\\s*;/g, 'var $1 = [];');
  src = src.replace(/\\(\\s*long\\s+long\\s*\\)/g, '');
  src = src.replace(/\\b(long\\s+long|int\\s*\\*|int|double)\\s+/g, 'var ');
  src = src.replace(/\\(\\*([A-Za-z_][A-Za-z0-9_]*)\\)/g, '$1[0]');
  src = src.replace(/\\*([A-Za-z_][A-Za-z0-9_]*)/g, '$1[0]');
  var addrMatches = src.match(/&([A-Za-z_][A-Za-z0-9_]*)/g) || [];
  var seen = {{}};
  for(var k=0;k<addrMatches.length;k++){{
    var v = addrMatches[k].slice(1);
    if(seen[v]) continue;
    seen[v] = 1;
    src = src.replace(new RegExp('\\\\bvar\\\\s+' + v + '\\\\s*=\\\\s*([^;]+);', 'g'), 'var ' + v + ' = [$1];');
    src = src.replace(new RegExp('&' + v + '\\\\b', 'g'), v);
    src = src.replace(new RegExp('\\\\b' + v + '\\\\b(?!\\\\s*\\\\[)(?!\\\\s*=)', 'g'), v + '[0]');
    src = src.split('var ' + v + '[0]').join('var ' + v);
    src = src.replace(new RegExp('(step_ada|write_ring)\\\\(([^)]*)\\\\b' + v + '\\\\[0\\\\]', 'g'), '$1($2' + v);
    src = src.replace(new RegExp('(step_ada|write_ring)\\\\(([^)]*)\\\\b' + v + '\\\\[0\\\\]', 'g'), '$1($2' + v);
  }}
  src = src.replace(/\\bNULL\\b/g, 'null');
  /* Preserve C++ integer division semantics in JS for page rounding and overflow bounds */
  src = src.replace(/\\(\\s*\\(\\s*req_bytes\\s*\\+\\s*page_size\\s*-\\s*1\\s*\\)\\s*\\/\\s*page_size\\s*\\)/g, 'Math.floor((req_bytes + page_size - 1) / page_size)');
  src = src.replace(/uint32_max\\s*\\/\\s*elem_bytes/g, 'Math.floor(uint32_max / elem_bytes)');
  return src;
}}

function wrapCppAdapter(ex, fn){{
  var id = ex.id;
  return function(){{
    var a = Array.prototype.slice.call(arguments);
    if(id === 'null_guard'){{
      var buf = [];
      var rc = fn(a[0], a[1], a[2], a[3], buf);
      return rc === 1 ? buf.slice(0, a[0]) : null;
    }}
    if(id === 'compound_guard'){{
      var st = [0,0,0];
      fn(a[0], a[0].length, a[1], st);
      return [st[0] === 1, st[1], st[2]];
    }}
    if(id === 'reassigned_leak'){{
      var r3 = [0,0,0];
      fn(a[0], a[1], a[2], a[3], r3);
      return r3;
    }}
    if(id === 'early_exit_uaf'){{
      var r3u = [0,0,0];
      fn(a[0], a[1], r3u);
      return r3u;
    }}
    if(id === 'loop_scratch_leak'){{
      var r2 = [0,0];
      fn(a[0], a[0].length, r2);
      return r2;
    }}
    if(id === 'tensor_view_uaf'){{
      var sl = [];
      fn(a[0], a[0].length, a[1], a[2], a[3], a[4], a[5], sl);
      if(sl.indexOf(-999999) >= 0) return sl;
      return sl.slice(0, a[5]);
    }}
    if(id === 'alloc_overflow'){{
      /* Simulate 32-bit unsigned bitmask cleanly in JS using Math.imul / modulo */
      return fn(a[0], a[1], a[2], a[3]);
    }}
    if(id === 'kv_dtype_memcpy'){{
      var dtMap = {{'fp32':0, 'fp16':1, 'bf16':2}};
      var cv = a[0].slice();
      var rc2 = fn(cv, dtMap[a[1]], a[2], dtMap[a[3]], a[4]);
      if(rc2 === -999) return 'ABORT_BYTE_WIDTH_MISMATCH';
      if(rc2 === -888) return 'CORRUPT_RAW_BITS';
      return cv;
    }}
    if(id === 'inplace_counter_alias'){{
      var pos = [], ncl = [0];
      fn(a[0][0], a[1], pos, ncl);
      return [pos, ncl[0]];
    }}
    if(id === 'persistent_buffer_state'){{
      var op = [0], os = [0];
      fn(a[0], a[1], a[2], a[3], a[4], op, os);
      return [Math.round(op[0]*1e6)/1e6, os[0]];
    }}
    if(id === 'cuda_pool_alloc'){{
      var cr = [0,0,0];
      fn(a[0], a[0].length, a[1], a[2], a[3], cr);
      return cr;
    }}
    if(id === 'autograd_graph_leak'){{
      var al = [0], lv = [0];
      fn(a[0], a[0].length, a[1], al, lv);
      return [Math.round(al[0]*1e6)/1e6, lv[0]];
    }}
    if(id === 'stream_record_lifetime'){{
      var sr = [0,0];
      fn(a[0], a[1], a[2], sr);
      return [sr[0], sr[1] === 1];
    }}
    if(id === 'pinned_alloc_cleanup'){{
      var pr = [0,0];
      fn(a[0], a[1], a[2], a[3], pr);
      return pr;
    }}
    if(id === 'dynamic_cache_reset'){{
      var ol = [0], ok = [];
      fn(a[0], a[0].length, a[1], a[1].length, a[2]?1:0, ol, ok);
      return [ol[0], ok.slice(0, ol[0])];
    }}
    if(id === 'ring_buffer_state_dict'){{
      var ob = [], osum = [0];
      fn(a[0], a[1], a[1].length, a[2], a[2].length, ob, osum);
      return [ob.slice(0, a[0]), osum[0]];
    }}
    return fn.apply(null, a);
  }};
}}

function eqVal(a, b){{
  if(typeof a === 'boolean' || typeof b === 'boolean') return a === b;
  if(typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(b));
  if(Array.isArray(a) && Array.isArray(b)){{
    if(a.length !== b.length) return false;
    for(var i=0;i<a.length;i++) if(!eqVal(a[i], b[i])) return false;
    return true;
  }}
  return a === b;
}}

function fmtVal(v){{
  if(v === null) return 'None / NULL';
  if(v === undefined) return 'undefined';
  try{{ return JSON.stringify(v); }}catch(e){{ return String(v); }}
}}

function pyLitToJs(s){{
  return s.replace(/\\bNone\\b/g, 'null').replace(/\\bTrue\\b/g, 'true').replace(/\\bFalse\\b/g, 'false');
}}

function gradeFn(fn, tests){{
  var results = [], allPass = true;
  for(var i=0;i<tests.length;i++){{
    var t = tests[i], r = {{hidden: !!t.h, pass: false, a: t.a}};
    try{{
      var args = (new Function('return (' + pyLitToJs(t.a) + ')'))();
      var exp = (new Function('return (' + pyLitToJs(t.e) + ')'))();
      r.e = fmtVal(exp);
      var got = fn.apply(null, args);
      r.got = fmtVal(got);
      r.pass = eqVal(got, exp);
    }}catch(err){{
      r.err = String(err && err.message || err);
    }}
    if(!r.pass) allPass = false;
    results.push(r);
  }}
  return {{ok: allPass, err: null, results: results}};
}}

function runExercise(ex, code, cb){{
  if(S.lang === 'cpp'){{
    try{{
      /* Fix 32-bit bitwise `& uint32_max` in JS where `|` / `&` are signed 32-bit: use modulo `(uint32_max + 1)` */
      var jsSrc = transpileCppToJsBrowser(code).replace(/\\(([^)]+\\*[^)]+)\\)\\s*&\\s*uint32_max/g, '(($1) % (uint32_max + 1))');
      var rawFn = (new Function(jsSrc + '\\nreturn ' + ex.cpp_name + ';'))();
      var wrapped = wrapCppAdapter(ex, rawFn);
      cb(gradeFn(wrapped, ex.tests));
    }}catch(err){{
      cb({{ok:false, err:'C++ compile/runtime error: ' + String(err && err.message || err), results:[]}});
    }}
    return;
  }}
  /* Python mode: try fast zero-latency in-page evaluator first; fall back to Pyodide Web Worker if custom Python syntax is used */
  try{{
    var fastCode = code.replace(/\\(([^)]+\\*[^)]+)\\)\\s*&\\s*uint32_max/g, '(($1) % (uint32_max + 1))');
    cb(runPyFastJs(ex, fastCode));
  }}catch(fastErr){{
    ensurePy();
    if(PYERR){{
      cb({{ok:false, err:'Python error: ' + String(fastErr && fastErr.message || fastErr), results:[]}});
      return;
    }}
    var pyScript = "import json\\n" + code + "\\n_res = []\\n_ok = True\\nfor _t in " + JSON.stringify(ex.tests) + ":\\n" +
      "    try:\\n" +
      "        _a = eval(_t['a'])\\n" +
      "        _e = eval(_t['e'])\\n" +
      "        _g = " + ex.py_name + "(*_a)\\n" +
      "        _p = (_g == _e)\\n" +
      "        if not _p: _ok = False\\n" +
      "        _res.append({{'hidden': bool(_t.get('h')), 'pass': bool(_p), 'a': _t['a'], 'e': json.dumps(_e), 'got': json.dumps(_g)}})\\n" +
      "    except Exception as _ex:\\n" +
      "        _ok = False\\n" +
      "        _res.append({{'hidden': bool(_t.get('h')), 'pass': False, 'a': _t['a'], 'e': _t['e'], 'err': str(_ex)}})\\n" +
      "json.dumps({{'ok': _ok, 'err': None, 'results': _res}})\\n";
    PYQ.push({{
      script: pyScript,
      cb: function(d){{
        if(d.fallback || d.err) cb({{ok:false, err:'Python error: ' + String(d.err || fastErr), results:[]}});
        else cb(JSON.parse(d.res));
      }}
    }});
    pyNext();
  }}
}}

function fmtResults(r){{
  if(r.err)return '<div class="res f">'+esc(r.err)+'</div>';
  var h='',p=0;
  for(var i=0;i<r.results.length;i++){{
    var t=r.results[i];if(t.pass)p++;
    if(t.pass)h+='<div class="res p">&#10003; '+(t.hidden?'hidden edge-case test':'test')+' '+(i+1)+' passed</div>';
    else if(t.hidden)h+='<div class="res f">&#10007; Hidden test '+(i+1)+' failed (checks boundary conditions like OOM, 0 items, or 32-bit wraparound).</div>';
    else h+='<div class="res f">&#10007; Test '+(i+1)+' failed:<br>&nbsp;&nbsp;input: <code>'+esc(t.a)+'</code><br>&nbsp;&nbsp;expected: <code>'+esc(t.e)+'</code><br>&nbsp;&nbsp;you got: <code>'+esc(t.err?('CRASH: '+t.err):t.got)+'</code></div>';
  }}
  return '<div class="res"><b>'+p+'/'+r.results.length+' tests pass</b></div>'+h;
}}

function renderCode(){{
  if(S.view && S.view !== 'list'){{
    var e = exById(S.view);
    if(e) return renderEx(e);
  }}
  var h = header() + '<div class="card small">Switch freely between <b>Python</b> and <b>C++</b> on any exercise. Every exercise starts with a real low-level memory, GPU allocator, or KV-cache bug that FAILS the tests (RED). Fix the bug to turn all visible and hidden tests GREEN, and inspect every line of code and every variable below!' + langBar() + '</div>';
  var nx = nextEx();
  if(nx) h += '<button class="go" data-ex="'+nx.id+'">Continue Next Unsolved Lab: '+esc(nx.title)+'</button>';
  for(var l=1;l<=NLEV;l++){{
    var n=0,d=0;
    h += '<div class="card"><b>Level '+l+': '+esc(L[l-1])+'</b>';
    for(var i=0;i<EX.length;i++){{
      if(EX[i].l !== l) continue;
      n++;
      var ok = exSolved(EX[i].id);
      if(ok) d++;
      h += '<button data-ex="'+EX[i].id+'">'+(ok?'&#10003; ':'&#9654; ')+esc(EX[i].title)+' <span class="tag bug">'+esc(EX[i].real_bug.split(' ')[0])+'</span></button>';
    }}
    h += '<div class="small">'+d+'/'+n+' solved in this level</div></div>';
  }}
  app.innerHTML = h;
  wireTabs();
  wireLang(renderCode);
  var bs = app.querySelectorAll('button[data-ex]');
  for(var k=0;k<bs.length;k++){{
    (function(b){{b.onclick=function(){{S.view=b.getAttribute('data-ex');save();render();window.scrollTo(0,0);}};}})(bs[k]);
  }}
}}

function renderEx(ex){{
  var isPy = (S.lang === 'py');
  var dk = ex.id + ':' + S.lang;
  var st = S.code[ex.id] || {{ok:0, tries:0, peek:0}};
  var starter = isPy ? ex.py_starter : ex.cpp_starter;
  var sol = isPy ? ex.py_sol : ex.cpp_sol;
  var draft = (S.drafts && S.drafts[dk] !== undefined) ? S.drafts[dk] : starter;

  var h = header() + '<div class="card">' +
    '<div class="small"><span class="tag">Level '+ex.l+'</span><span class="tag bug">'+esc(ex.real_bug)+'</span>'+(st.ok?'<span class="tag" style="color:#4cc38a">&#10003; solved</span>':'')+'</div>' +
    langBar() +
    '<p class="q">'+esc(ex.title)+'</p>' +
    '<p style="white-space:pre-line">'+esc(ex.desc)+'</p>' +
    '<textarea id="ta" autocapitalize="off" autocorrect="off" autocomplete="off" spellcheck="false" wrap="off"></textarea>' +
    '<div class="row"><button class="go" id="run">Run Tests ('+(isPy?'Python':'C++')+')</button><button id="hint">Hint</button></div>' +
    '<div class="row"><button id="show">Show Answer &amp; Full Line-by-Line Guide</button><button id="reset">Reset to Buggy Starter</button></div>' +
    '<div id="out"></div>' +
    '<div style="margin-top:12px">' + explainHtml(ex, false) + '</div>' +
    '</div><button id="back">Back to All 16 Code Labs</button>';
  app.innerHTML = h;
  wireTabs();
  wireLang(function(){{renderEx(ex);}});
  var m = document.getElementById('menu'); if(m) m.onclick = menu;
  var ta = document.getElementById('ta');
  ta.value = draft;
  ta.oninput = function(){{if(!S.drafts)S.drafts={{}};S.drafts[dk]=ta.value;save();}};
  var out = document.getElementById('out');
  document.getElementById('run').onclick = function(){{
    runExercise(ex, ta.value, function(r){{
      var c = S.code[ex.id] || {{ok:0, tries:0, peek:0}};
      c.tries = (c.tries||0) + 1;
      var msg = fmtResults(r);
      if(r.ok){{
        c.ok = 1; S.code[ex.id] = c; S.stats.ans++; S.stats.right++;
        msg = '<div class="why"><b>All visible &amp; hidden tests pass!</b> You fixed the bug cleanly.</div>' + msg +
          '<button class="go" id="nx">'+(nextEx()?'Next Unsolved Exercise':'Back to Exercise List')+'</button>';
      }} else {{
        S.code[ex.id] = c;
        msg += '<div class="small" style="margin-top:6px">Remember: the starter code has the real bug so it fails on purpose! Tap <b>Hint</b> or expand <b>1. What Every Variable Means</b> &amp; <b>2. Line-by-Line Code Walkthrough</b> below.</div>';
      }}
      out.innerHTML = msg;
      save();
      var nx = document.getElementById('nx');
      if(nx) nx.onclick = function(){{var n=nextEx();S.view=n?n.id:'list';save();render();window.scrollTo(0,0);}};
    }});
  }};
  document.getElementById('hint').onclick = function(){{
    out.innerHTML = '<div class="why"><b>Hint:</b> '+esc(ex.hint)+'</div>' + out.innerHTML;
  }};
  document.getElementById('show').onclick = function(){{
    var c = S.code[ex.id] || {{ok:0, tries:0, peek:0}}; c.peek = 1; S.code[ex.id] = c; save();
    out.innerHTML = '<div class="why idk"><b>Reference Fix ('+(isPy?'Python':'C++')+'):</b><pre class="sol">'+esc(sol)+'</pre>' +
      '<button id="copyfix" class="go" style="margin-top:6px">Load Fix Into Editor &amp; Run</button></div>' + explainHtml(ex, true);
    var cf = document.getElementById('copyfix');
    if(cf) cf.onclick = function(){{ ta.value = sol; ta.oninput(); document.getElementById('run').onclick(); }};
  }};
  document.getElementById('reset').onclick = function(){{
    ta.value = starter; ta.oninput(); out.innerHTML = '';
  }};
  document.getElementById('back').onclick = function(){{
    S.view = 'list'; save(); render();
  }};
}}

function renderGuide(){{
  var h = header() + '<div class="card"><b>Complete Interactive Reference &amp; Study Guide (All 8 Levels, 16 C++ &amp; Python Fixes)</b>' +
    '<div class="small" style="margin-top:4px">Every exercise below shows both the buggy starter and the verified fix in C++ and Python, with every variable and every single line of code explained.</div></div>';
  for(var i=0;i<EX.length;i++){{
    var ex = EX[i];
    h += '<div class="card">' +
      '<div class="small"><span class="tag">Level '+ex.l+': '+esc(L[ex.l-1])+'</span><span class="tag bug">'+esc(ex.real_bug)+'</span></div>' +
      '<p class="q">'+esc(ex.title)+'</p>' +
      '<p style="white-space:pre-line">'+esc(ex.desc)+'</p>' +
      '<details open class="why"><summary><b>Python Fix (Reference Solution)</b></summary><pre class="sol">'+esc(ex.py_sol)+'</pre></details>' +
      '<details open class="why"><summary><b>C++ Fix (Reference Solution)</b></summary><pre class="sol">'+esc(ex.cpp_sol)+'</pre></details>' +
      explainHtml(ex, true) +
      '<button class="go" data-goex="'+ex.id+'">Open This Lab in the Interactive Code Runner</button>' +
    '</div>';
  }}
  app.innerHTML = h;
  wireTabs();
  var m = document.getElementById('menu'); if(m) m.onclick = menu;
  var bs = app.querySelectorAll('button[data-goex]');
  for(var k=0;k<bs.length;k++){{
    (function(b){{b.onclick=function(){{S.mode='code';S.view=b.getAttribute('data-goex');save();render();window.scrollTo(0,0);}};}})(bs[k]);
  }}
}}

function schedule(id,gap){{S.due.push({{id:id,at:S.n+gap}});}}

function checkLevelUp(){{
  if(S.level<NLEV&&mastered(S.level)>=PER-1){{
    S.level++;if(S.level>S.top)S.top=S.level;
    return 'Level up! Now Level '+S.level+': '+esc(L[S.level-1])+'.';
  }}
  if(S.level===NLEV&&mastered(NLEV)>=PER-1&&!S.done){{
    S.done=true;return 'You mastered all '+NLEV+' levels! Questions now mix across all levels for spaced repetition.';
  }}
  return '';
}}

function answer(o){{
  if(answered)return;answered=true;
  var q=BANK[cur.id],id=cur.id,b=bx(id),good=(o===0),idk=(o===-1);
  S.n++;S.stats.ans++;
  var btns=document.querySelectorAll('#opts button'),i;
  for(i=0;i<btns.length;i++){{
    var v=parseInt(btns[i].getAttribute('data-o'),10);
    btns[i].disabled=true;
    if(v===0)btns[i].className+=' ok';
    else if(v===o&&!good)btns[i].className+=' bad';
  }}
  var msg,cls='';
  if(good){{
    S.stats.right++;
    if(b<0)S.box[id]=2;else S.box[id]=Math.min(9,b+1);
    if(b>=0&&b<2)schedule(id,10);
    msg='Correct! ';
  }}else{{
    S.box[id]=0;schedule(id,3);
    if(idk){{S.stats.idk++;cls=' idk';msg='No problem &mdash; here is how it works: ';}}
    else{{cls=' bad';msg='Not quite. ';}}
    msg+='The right answer is: <b>'+esc(q.o[0])+'</b>.<br><br>';
  }}
  var extra=!good?'<div class="small" style="margin-top:6px">Scheduled for spaced-repetition review in 3 questions.</div>':'';
  var lvlmsg=checkLevelUp();
  document.getElementById('fb').innerHTML='<div class="why'+cls+'">'+msg+esc(q.w)+extra+'</div>'+
    (lvlmsg?'<div class="why" style="margin-top:8px"><b>'+lvlmsg+'</b></div>':'')+
    '<button class="go" id="next">Next Question</button>';
  document.getElementById('next').onclick=function(){{cur=null;save();render();}};
  save();
}}

function menu(){{
  var h='<h1>All 8 Ladder Levels</h1><div class="card">',i;
  for(i=1;i<=NLEV;i++){{
    h+='<div style="margin:6px 0">'+(i===S.level?'&#9654; ':'')+'<b>'+i+'.</b> '+esc(L[i-1])+' <span class="small">('+mastered(i)+'/'+PER+' quiz mastered)</span></div>';
  }}
  h+='</div><div class="card small">Jump directly to any level: <select id="jmp">';
  for(i=1;i<=NLEV;i++)h+='<option value="'+i+'"'+(i===S.level?' selected':'')+'>'+escapeHtml(L[i-1])+'</option>';
  h+='</select><button class="go" id="go">Jump to Level</button><button id="rst">Reset All Progress</button></div><button class="go" id="back">Back</button>';
  app.innerHTML=h;
  document.getElementById('go').onclick=function(){{S.level=parseInt(document.getElementById('jmp').value,10);cur=null;save();render();}};
  document.getElementById('back').onclick=function(){{render();}};
  document.getElementById('rst').onclick=function(){{
    if(confirm('Reset all quiz and code progress?')){{
      S={{level:1,top:1,box:{{}},due:[],n:0,stats:{{ans:0,right:0,idk:0}},done:false,code:{{}},drafts:{{}},mode:'quiz',view:'list',lang:'py'}};
      cur=null;save();render();
    }}
  }};
}}
function escapeHtml(s){{return esc(s);}}

function render(){{
  if(S.mode==='code')return renderCode();
  if(S.mode==='guide')return renderGuide();
  if(!cur){{var id=pick();cur={{id:id,order:shuffle([0,1,2,3])}};answered=false;}}
  var q=BANK[cur.id],h=header();
  var again=(bx(cur.id)>=0&&bx(cur.id)<2)?'<span class="tag">spaced review</span>':'';
  h+='<div class="card"><div class="small">'+again+'<span class="tag">Level '+q.l+': '+esc(L[q.l-1])+'</span></div><p class="q">'+esc(q.q)+'</p><div id="opts">';
  for(var i=0;i<4;i++)h+='<button data-o="'+cur.order[i]+'">'+esc(q.o[cur.order[i]])+'</button>';
  h+='<button class="idk" data-o="-1">I don\\\'t know yet (explain it to me in plain words)</button></div><div id="fb"></div></div>';
  app.innerHTML=h;
  var b=document.querySelectorAll('#opts button');
  for(var k=0;k<b.length;k++){{(function(el){{el.onclick=function(){{answer(parseInt(el.getAttribute('data-o'),10));}};}})(b[k]);}}
  var m=document.getElementById('menu');if(m)m.onclick=menu;
  wireTabs();
}}

load();
if(!S.code)S.code={{}};if(!S.drafts)S.drafts={{}};if(!S.mode)S.mode='quiz';if(!S.view)S.view='list';if(!S.lang)S.lang='py';
render();
}})();
</script></body></html>
"""
    return html


def build_markdown_guide() -> str:
    """Builds the comprehensive Markdown Study Guide with all 8 levels, 64 questions, and 16 C++/Python exercises."""
    md = []
    md.append("# Advanced ML Deep-System Low-Level Memory, GPU Allocator & Cache Fixes — Interactive Ladder Study Guide\n")
    md.append("> [!IMPORTANT]")
    md.append("> **How to use this Study Guide:**")
    md.append("> 1. **Interactive Web App (`ml_memory_ladder.html`):** Open `/home/droid/ml-memory-ladder/ml_memory_ladder.html` in any browser (desktop or mobile Safari). It follows the exact **ML Ladder** pattern (`Quiz` + `Code Labs` in both **Python** and **C++** + `Full Study Guide` tab).")
    md.append("> 2. **Grounded in Your Real Bug Ledger:** Every level and code lab is tied directly to real bugs you found, verified, or audited across `deepspeedai/DeepSpeed#8565`, `cfengine/core#6358`, `eclipse-mosquitto/mosquitto#3762`, `pytorch/executorch#23399`, `huggingface/transformers#48509/#48809`, `GPT-Neo StaticCache`, `Kornia ADA`, and `TorchMetrics Running` / `WrapperMetric`.")
    md.append("> 3. **Every Variable & Every Line Explained:** For all 16 exercises across 8 levels, every single variable, pointer, tensor metadata field, C++ line, and Python line is documented below.\n")

    md.append("## Ladder Architecture Overview\n")
    md.append("```mermaid")
    md.append("flowchart TD")
    md.append('    L1["Level 1: Pointers, Addresses & NULL Checks (CWE-476)"] --> L2["Level 2: Heap Lifetimes & Memory Leaks (CWE-401)"]')
    md.append('    L2 --> L3["Level 3: Dangling Pointers, UAF & Double-Free (CWE-416/415)"]')
    md.append('    L3 --> L4["Level 4: Allocation Math (IO2BO CWE-190) & Raw memcpy Dtype Traps"]')
    md.append('    L4 --> L5["Level 5: Python/PyTorch Views, In-Place Aliasing & Persistent Buffers"]')
    md.append('    L5 --> L6["Level 6: GPU CUDACachingAllocator Splitting, OOM Retry & Autograd Leaks"]')
    md.append('    L6 --> L7["Level 7: Async CUDA Streams (record_stream) & Pinned Host Memory (mlock)"]')
    md.append('    L7 --> L8["Level 8: LLM KV-Cache (Dynamic vs Static) & Ring-Buffer Lifetimes"]')
    md.append("```\n")

    for lvl_idx, lvl_title in enumerate(LEVELS, start=1):
        md.append(f"---\n\n## {lvl_title}\n")
        qs = [q for q in QUIZ if q["l"] == lvl_idx]
        md.append(f"### Level {lvl_idx} Self-Check Quiz ({len(qs)} Questions)\n")
        for qi, q in enumerate(qs, start=1):
            md.append(f"**Q{lvl_idx}.{qi}: {q['q']}**")
            md.append(f"- **Correct Answer:** {q['o'][0]}")
            md.append(f"- **Why / Systems Mechanism:** {q['w']}\n")

        exs = [e for e in EXERCISES if e["l"] == lvl_idx]
        for ex in exs:
            md.append(f"### Lab {ex['title']}\n")
            md.append(f"- **Real Bug Origin:** `{ex['real_bug']}`\n")
            md.append(f"{ex['desc']}\n")

            md.append("#### Every Variable & Parameter Explained\n")
            md.append("| Variable / Parameter | Exact Role in Hardware / RAM / VRAM & Failure Mode |")
            md.append("|---|---|")
            for vname, vdesc in ex["v"]:
                md.append(f"| `{vname}` | {vdesc} |")
            md.append("")

            md.append("#### C++ Buggy Starter vs. Verified Fix\n")
            md.append("**Buggy C++ Starter (RED — Fails Tests):**\n```cpp\n" + ex["cpp_starter"].strip() + "\n```\n")
            md.append("**Fixed C++ Solution (GREEN — Passes All Tests):**\n```cpp\n" + ex["cpp_sol"].strip() + "\n```\n")

            md.append("#### C++ Line-by-Line Walkthrough\n")
            md.append("| C++ Code Line | Detailed Low-Level Explanation |")
            md.append("|---|---|")
            for c_line, c_exp in ex["lines_cpp"]:
                safe_line = c_line.replace("|", "\\|")
                md.append(f"| `{safe_line}` | {c_exp} |")
            md.append("")

            md.append("#### Python Buggy Starter vs. Verified Fix\n")
            md.append("**Buggy Python Starter (RED — Fails Tests):**\n```python\n" + ex["py_starter"].strip() + "\n```\n")
            md.append("**Fixed Python Solution (GREEN — Passes All Tests):**\n```python\n" + ex["py_sol"].strip() + "\n```\n")

            md.append("#### Python Line-by-Line Walkthrough\n")
            md.append("| Python Code Line | Detailed Low-Level Explanation |")
            md.append("|---|---|")
            for p_line, p_exp in ex["lines_py"]:
                safe_pline = p_line.replace("|", "\\|")
                md.append(f"| `{safe_pline}` | {p_exp} |")
            md.append("")

    return "\n".join(md)


def main():
    out_dir = "/home/droid/ml-memory-ladder"
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, "ml_memory_ladder.html")
    index_path = os.path.join(out_dir, "index.html")
    md_path = os.path.join(out_dir, "STUDY_GUIDE.md")

    html_content = build_html()
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    md_content = build_markdown_guide()
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Built {html_path} ({len(html_content)} bytes)")
    print(f"Built {md_path} ({len(md_content)} bytes)")


if __name__ == "__main__":
    main()
