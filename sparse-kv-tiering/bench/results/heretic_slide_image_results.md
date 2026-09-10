# Sliding window with an image in the conversation

A = full prefill (reference), B = the same minus its first third, slid onto A's cache.

== kept ==
kept_A: prompt_n 32717, 121.3s, 34.9 t/s, needles 3/4, image no | 'Based on the text provided, here are the facts extracted from the text:\n\n**Fact '
kept_B: prompt_n 49, 0.9s, 37.9 t/s, needles 4/4, image no | 'Based on the text provided, here are the facts extracted from the text:\n\n**Fact '
shifts 1, forced re-prefill 0, reuse refused 0
== dropped ==
dropped_A: prompt_n 32717, 139.4s, 21.9 t/s, needles 2/4, image read | 'Based on the text provided, here are the extracted facts corresponding to the ma'
dropped_B: prompt_n 39, 0.9s, 32.4 t/s, needles 2/4, image no | '1. plate tectonics recycles crust at subduction zones and creates it at mid-ocea'
shifts 1, forced re-prefill 0, reuse refused 0
== newturn ==
newturn_A: prompt_n 32607, 120.5s, 42.6 t/s, needles 2/4, image no | '1. Fact 80: plate tectonics recycles crust at subduction zones and creates it at'
newturn_B: prompt_n 151, 2.2s, 35.7 t/s, needles 2/4, image no | '1. plate tectonics recycles crust at subduction zones and creates it at mid-ocea'
shifts 1, forced re-prefill 0, reuse refused 0
