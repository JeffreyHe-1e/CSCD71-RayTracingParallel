15 minutes (3-5 mins)
### Need the following
- Introduction (people might not know what my problem is)
- Program approach
- Parallelism 
	- MPI and OpenACC
	- Profiling, and comparison with bench mark.
	- Serialization Analysis

### New code
``` pseudocode
foreach pixel (i,j):
	create rays (randomize directions for anti-aliasing)
	foreach ray:
		foreach recursionDepth:
			find-first-hit
			
		end foreach recursionDepth
	end foreach ray
end foreach pixel
```
