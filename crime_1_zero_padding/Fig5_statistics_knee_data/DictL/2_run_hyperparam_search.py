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

The advantage of the above approach is that these runs can be sent in parallel to many CPUs, on different servers.

Tip:
In order to send 20 runs in parallel (i.e. to 20 CPUs on one server), run this (in the linux command line):
cat run_xxxxxx.sh | xargs -n1 -I{} -P20 bash -c {} > log.txt

Notice that this huge set of runs is expected to take a VERY LONG TIME! In our lab it was conducted over 200 CPUs in
parallel, and it required about 4 weeks.

(c) Efrat Shimron (UC Berkeley) & Jon Tamir (UT Austin) (2021)

'''

