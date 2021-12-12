'''
Instructions - how to calibrate the DictL parameters by sending many parallel runs:
1. gen_commands.sh - edit this script using linux only, DO NOT EDIT IT IN WINDOWS/PYCHARM! It doesn't compile afterwards.
2. make it excutable by running:
   chomd +x gen_commands.sh
3. run it:
   ./gen_commands.sh
   This will create several scripts named run_xxxxxxxx.sh
4. The script print_timestamp is also needed for the grid search. Make it executable by running:
   chmod +x print_timestamp.sh
5. Send each of the run_xxxx.sh scripts manually. Notice: each of them contains dozens of python runs. Every set of
   runs can take a very long time.

(c) Efrat Shimron (UC Berkeley, 2021)

'''

