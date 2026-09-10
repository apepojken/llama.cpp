# Sliding window (middle eviction + KV shift) checks

- **B**: prompt tokens processed 22 of 46944 evaluated (cached 46975), prompt time 1.3 s, decode 30.8 t/s, draft accept 0.85; first line correct: False; answer: '</think>\n\n### Task: quote the very first line of the text above, verbatim, and n'
- **C**: prompt tokens processed 23483 of 23483 evaluated (cached 23514), prompt time 68.8 s, decode 22.7 t/s, draft accept 0.53; first line correct: False; answer: '</parameter>\n<parameter=parameter=verbatim>\n</parameter>\n</function>\n</tool_call'
- **Bctl**: prompt tokens processed 46944 of 46944 evaluated (cached 46975), prompt time 174.0 s, decode 29.5 t/s, draft accept 0.75; first line correct: False; answer: '</parameter>\n</function>\n</tool_call>\n\n<tool_call>\n<function=Bash>\n<parameter=co'
- slide vs control answers differ (legitimately may differ: the slid state keeps a trace of the dropped text)
