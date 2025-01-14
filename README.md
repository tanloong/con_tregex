Tregex is a Java program for identifying patterns in constituency trees. PyTregex is a Python implementation of Tregex.

## Basic usage

### Command-line

To use the command-line version, simply use `pip install` to install it, and then run it using `python -m pytregex`.

```sh
$ pip install pytregex

$ echo '(NP(DT The)(NN battery)(NN plant))' | python -m pytregex pattern 'NP < NN' -filter
(NP (DT The) (NN battery) (NN plant))
(NP (DT The) (NN battery) (NN plant))
There were 2 matches in total.

$ echo '(NP(DT The)(NN battery)(NN plant))' > trees.txt
$ python -m pytregex pattern 'NP < NN' ./trees.txt
(NP (DT The) (NN battery) (NN plant))
(NP (DT The) (NN battery) (NN plant))
There were 2 matches in total.

$ python -m pytregex pattern 'NP < NN' -C ./trees.txt
2

$ python -m pytregex pattern 'NP < NN=a' -h a ./trees.txt
(NN battery)
(NN plant)
There were 2 matches in total.

$ python -m pytregex explain '<'
'A < B' means A immediately dominates B

$ python -m pytregex pprint '(NP(DT The)(NN battery)(NN plant))'
NP
├── DT
│   └── The
├── NN
│   └── battery
└── NN
    └── plant
```

### Inline

```python
from pytregex.tregex import TregexPattern

tre = TregexPattern("NP < NN=a")
matches = tre.findall("(NP(DT The)(NN battery)(NN plant))")
handles = tre.get_nodes("a")
print("matches nodes:\n{}\n".format("\n".join(str(m) for m in matches)))
print("named nodes:\n{}".format("\n".join(str(h) for h in handles)))

# Output:
# matches nodes:
# (NP (DT The) (NN battery) (NN plant))
# (NP (DT The) (NN battery) (NN plant))
#
# named nodes:
# (NN battery)
# (NN plant)
```

## Differences from Tregex
### ; |
### backward referencing not supported
### regex naming?
