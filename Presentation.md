15 minutes (3-5 mins)
### Need the following
- Introduction (people might not know what my problem is)
- Program approach
- Parallelism 
	- [x] OpenMP
	- [ ] MPI 
	- [ ] OpenACC
	- Profiling, and comparison with bench mark.
	- Serialization Analysis
# tasks #TODO
- [ ] generate scene
	- [ ] make trunk 
	- [ ] make branch fractal
	- [ ] make leaves
	- [ ] have variations
- [ ] optimize
	- [ ] turn recursion to for loops
	- [ ] turn object linked list to array so many threads can operate on it
	- [ ] generate lists of random values
	- [ ] 
### New code
``` pseudocode
foreach pixel (i,j):
	create rays (randomize directions for anti-aliasing)
	foreach ray:
		foreach recursionDepth:
			find-first-hit
			if hit then shade 
			else shade with background colour
		end foreach recursionDepth
	end foreach ray
end foreach pixel
```

