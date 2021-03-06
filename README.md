# fundl

A totally unserious, pedagogical, functional-oriented deep learning library built on top of autograd.

You have been forewarned: this is a package that is designed to be educational, filled with experiments. **Do not use it in production!** You can, however, use it for a better understanding of how the internals of deep learning work.

## why this exists

**(1) Human reinforcement learning.**
I aim to implement *things* from the deep learning world here,
so that I and others can learn exactly what's going on underneath the hood.

**(2) Education.**
This code hopefully demystifies some of the internals of deep learning libraries.

**(3) Experimentation.**
I want to see what it takes to build a functional-first deep learning library.
It is also an experiment in thinking about deep learning algorithms as functions,
rather than objects.
Minor pedantic detail!

**(4) Compatibility with the PyData stack.**
The tensor libraries out there are sometimes just frustrating to use.
We can't easily plug their tensors into `matplotlib` and `pandas`,
mainly because they do not follow Python protocols.
It is my hope that this situation rectifies over time.
Until then, this library hopefully solves that problem.

## what will not be done

I explicitly strive to *not* replicate
Keras,
Pytorch
or Chainer.
If you need stability and reliability,
go use those libraries instead.

## how to contribute

If this project interests you,
ways you can contribute include
neural network layers (i.e. functions)
layers,
tests,
examples,
and/or docs.

### layers

If you contribute a neural network layer,
be sure to also add in docstrings and tests.
See the `tests` section for more details.

### tests

Tests should ensure that tensor shapes are correctly implemented.

### examples

They go inside the `examples/` directory.

### docs

Docstrings should exist for each function.
Because they are primarily to be rendered in an IDE
(e.g. Jupyter)
they should prioritize code readability
rather than LaTeX rendering.
