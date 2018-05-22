import pstats
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python printProfile.py profilename [cumtime|internal|all]")
        exit()
    filename = sys.argv[1]
    p = pstats.Stats(filename)
    if len(sys.argv) == 3:
        flag = sys.argv[2]
        if flag == 'cumtime':
            p.sort_stats('cumulative').print_stats(15)
        elif flag == 'all':
            p.sort_stats('cumulative').print_stats()
        elif flag == 'internal':
            p.sort_stats('time').print_stats(10)
        else:
            print('Invalid flag')
    else:
        p.sort_stats('cumulative').print_stats(15)
