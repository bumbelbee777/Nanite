# SillyAI + SillyISA Sketchbook

## What's This?

This is a sketchbook on implementation plans and notes for SillyAI and SillyISA that detail on how they will be coded and turned into something real-world and tangible and not just intriguing theoritical concepts.

## SillyAI

Currently, SillyAI is practically fully implemented. It just needs real-world testing and validation. It could use additional/better plugin support and a context window but that's not the priority right now.

## SillyVM

This is what needs the most work right now. Sure, I got it implemented. But is it ideal? Absolutely freaking not. Does it get the job done? Maybe (currently implementing variable declaration support).

First revision of spec is virtually complete, it just needs to be implemented right. For this I crafted a future rewrite strategy:

- `Opcodes` enum for opcodes to allow for instructions to be tweaked easily.
- Have a `ExpressionParser` class that parses/evaluates bytecode and expressions before organizing them into segments to then be loaded into memory easily (variable assignment).
- A `MemoryUnit` class that will handle memory ops and management as well as things like NaN-boxing and caching/prefetching.
- `InstructionPipeline` handles all the instruction pipelining stuff and async VLIW/EPIC.
- Have a `BytecodeEngine` class that does all the bytecode execution. Self-explanatory.
- Boilerplate variables, classes, and functions for type checking and evaluation (`TypedName` and `TypedValue` for instance).
- For data type parsing we'll use regexes to match them, saving lines of code and development complexity.
- All this will be orchestrated by a `SillyVM` class that handles all this for you. Just pass in your code, number of instances you want, and some other configuration info, and you're good to go.
- First milestone is to get a pythagorean theorem proof bytecode program to execute successfully. Then all the specs will be implemented gradually (with concept + logic stuff being last due to its complexity).