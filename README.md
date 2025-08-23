# HydroDaemonDemo

This repository contains the code associated with the article: "Banishing
Banishing ancient numerical daemons of conceptual hydrology: robust
hydrological simulators with Julia, DifferentialEquations.jl, and automatic
differentiation".

## Installation

The numerical code in this repository is organized as a Julia package. In case
you are unfamiliar with Julia, refer to the "Getting Started" section.

## Getting Started with Julia

### Getting Julia

On Windows, Julia may be installed from the Microsoft Store.

For other operation systems, you may generally run:

```console
curl -fsSL https://install.julialang.org | sh
```
Julia will be available via your command line interface after installation.

Alternatively, refer to the [Julia downloads page](https://julialang.org/downloads/).

If you use and prefer package managers, note that many Linux/BSD managers ship
broken or very old versions of Julia. The [pixi](https://pixi.sh/latest/)
package manager provides a modern Julia.

### Running Julia programs

You can start Julia from the command line by invoking:

```console
julia
```

This opens the REPL (read-evaluate-print-loop) and you'll be able to import
modules, run commands interactively, etc.

Generally, however, you want to edit and run scripts interactively via an
editor. Julia works well with [VSCode](https://code.visualstudio.com/). Install
the Julia VSCode extension.  

To run an example from this repository, open ``examples/basic.jl`` and execute
lines via ``Shift + Enter``. Running this example will automatically install
all the required packages and dependencies thanks to Julia's package manager.

### Multiple Dispatch

Julia's syntax is very readable, and you should be able to find your way in
this repository fairly easily. However, Julia is not an object oriented
language, but uses so-called multiple dispatch instead. Since multiple dispatch
is not a common feature of many languages, we will briefly introduce it here.

Where object oriented languages store data structures and methods (associated
functions) together in classes, Julia stores them separately. To associate
functions with data structures, we annotate the argument types.

We'll start with a simple Python toy example for a data structure containing an
array of numbers.

```python
class MyNumbers:
    def __init__(self, a):
        self.a = a

    def add(self, b):
        for i, number in enumerate(self.a):
             self.a[i] += b
        return

    def sum(self):
        accumulator = 0.0
        for number in self.a:
             accumulator += number
        return accumulator


numbers = MyNumbers(np.array([1.0, 2.0, 3.0]))
numbers.add(4.0)
numbers_sum = numbers.sum()
```

The Julia equivalent is:
```julia
struct MyNumbers
     a::Vector{Float64}
end

function add!(my_numbers::MyNumbers, b)
    for (i, number) in enumerate(my_numbers.a)
         my_numbers.a[i] += b
    end
    return
end

function sum(my_numbers::MyNumbers)
    accumulator = 0.0
    for number in my_numbers.a
         accumulator += number
    return accumulator
end

numbers = MyNumbers([1.0, 2.0, 3.0])
add!(numbers, 4.0)
numbers_sum = sum(numbers)
```

By specifying `::MyNumbers` on the argument, we tell Julia that this specific
function should be called when the argument type is `MyNumbers`. For example,
there may be many `sum` methods (one for integers, one for floats, etc.). Julia
determines that the type of the first argument is `MyNumbers` and finds the
appropriate function to call. The object oriented way is sometimes also called
single dispatch, because the first argument (`self`) determines which method is
called. The exclamation mark in `add!` is a Julia convention to mark methods
that mutate data (instead of returning a new object).

Let's illustrate multiple dispatch by adding another add method, which adds the
numbers of another `MyNumbers` instance. In Python, we have no other choice but
to define a new method with a separate name:

```python
def add_numbers(self, other_numbers):
    for i, (number_a, number_b) in enumerate(zip(self.a, other_numbers.a)):
         self.a[i] = number_a + number_b
    return

other = MyNumbers(np.array([4.0, 5.0, 6.0]))
numbers.add_numbers(other)
```

In Julia, we can simply define another `add!` method and annotate the second
argument as well:
```julia
function add!(my_numbers::MyNumbers, other_numbers::MyNumbers)
     for (i, (number_a, number_b)) in enumerate(zip(my_numbers.a, other_numbers.a))
          my_numbers.a[i] = number_a + number_b
     end
     return
end
```

Julia will now dispatch on the type of both arguments, making it **multiple**
dispatch! This means the same function name can have different implementations
based on the types of all its arguments, not just the first one.

This repository makes extensive use of multiple dispatch to share functionality
between different kinds of hydrological models. For example, each model needs
similar time stepping, but its water balance function depends on the type of
model (reservoirs, FUSE-070, or Richards equation).

## Package organization

This package provides three main data structures:

1. `ExplicitHydrologicalModel` for a hydrological model with first-order
   forward in time discretization (explicit).
2. `ImplicitHydrologicalModel` for a hydrological model with first-order
   backward in time discretization (implicit).
3. `DiffEqHydrologicalModel` for a hydrological model that uses a method-of-lines (MOL)
   approach via `DifferentialEquations.jl` and through it supports many (higher) order
   methods, explicit, implicit, and mixed formulations.

### Required methods

Defining a model requires:

* A `State` struct, containing the state variables,
  the time derivatives, and state-dependent variables.
* A `Parameters` struct which holds the physical
  parameters and the (meteorological) forcing time series.
* A `prepare_state` method which takes the parameters, an initial state vector, and the forcing and returns the appropriate `State` structure.

`ExplicitHydrologicalModel` is the simplest and requires only a `waterbalance!`
function which computes the derivative `du/dt`.

`ImplicitHydrologicalModel` is more involved, it requires:

* `waterbalance!`
* `residual!` (a wrapper around `waterbalance!`)
* `jacobian!`
* `copy_state!` to store the previous state.
* `rewind!` to reset the previous state in case of convergence failure.

Parameters and meteorological forcing are combined in a `Parameters` struct.
The mutable state and dependent variables are stored in a `State` struct.

Finally, `DiffEqHydrologicalModel` requires `waterbalance!` which is a
trivial wrapper around the bare `waterbalance!` and the parametric `DiffEqParams` type, ensuring
we dispatch on the type of hydrological model (encoded by the state and parameter structs).

Minor code duplication exists in the explicit, implicit, and
DifferentialEquations formulation of the right-hand-side (RHS) function of the
differential equation. A general `rhs!` function could serve all models, but at
some cost to readability; the current setup matches the code of most
hydrological models which generally support only one or a few formulations.

## Running a model

Running a model requires:

* loading the meteorological forcing timeseries, and creating `Forcing` from them,
* defining the model specific `Parameters`,
* specifying an initial state,
* specifying a time span,
* choosing a model type; 
* and configuring the solver for Implicit or DiffEq models.

## Memory allocation and performance

Julia is a garbage-collected languages without the need for manual memory management.
This makes it memory safe by default and easy to write, but this has the downside that
it's easy to slow down programs by accidental dynamic allocations and garbage collection
cycles. The code in this repository has been carefully written to minimize memory allocations.
