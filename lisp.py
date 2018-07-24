###############################################################################
# Helper code for parsing Racket/Lisp code
# mostly from http://norvig.com/lispy.html
#
# MIT License Copyright (c) 2010-2017 Peter Norvig
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
#
# Modifications by Lisa Zhang, Gregory Rosenblatt
###############################################################################

def parse(program):
    "Read a minikanren/racket expression from a string."
    #return read_from_tokens(tokenize(program))
    return read_from_tokens_stack(tokenize(program))

def read_from_tokens_stack(tokens):
    "Read an expression from a sequence of tokens, without recursion."
    if len(tokens) == 0:
        raise SyntaxError('unexpected EOF while reading')
    stack = [[]]
    for token in tokens:
        if '(' == token:
            stack.append([])
        elif ')' == token:
            a = stack.pop()
            stack[-1].append(a)
        else:
            # Inlining atom saves a couple seconds.
            if token[0].isdigit(): stack[-1].append(int(token))
            else: stack[-1].append(token)
    assert len(stack) == 1
    result = stack[0][0]
    return result

def read_from_tokens(tokens):
    "Read an expression from a sequence of tokens."
    if len(tokens) == 0:
        raise SyntaxError('unexpected EOF while reading')
    token = tokens.pop(0)
    if '(' == token:
        L = []
        while tokens[0] != ')':
            L.append(read_from_tokens(tokens))
        tokens.pop(0) # pop off ')'
        return L
    elif ')' == token:
        raise SyntaxError('unexpected )')
    else:
        return atom(token)

def atom(token):
    "Numbers become numbers; every other token is a symbol."
    if token[0].isdigit():
        return int(token)
    else:
        return token

def tokenize(chars):
    "Convert a string of characters into a list of tokens."
    tokens = chars.replace('(', ' ( ') \
            .replace(')', ' ) ') \
            .replace("'", "' ") \
            .split()
    return [t for t in tokens if t]

def unparse(ast, collapse_lvar=False):
    "Convert a Python representation of a lisp expression into a string"
    if type(ast) == list:
        return "(%s)" % ' '.join(unparse(child, collapse_lvar)
                                 for child in ast)
    ast = str(ast)
    if collapse_lvar and ast.startswith("_."):
        return "_"
    return ast

