import argparse
import os, sys
import random
import numpy as np
import time
random.seed(123)
initial_set = 4
begin2end = 3
iters = 100

from .executor import Executor, LOG_DIR

class GA:
    def __init__(self, options, get_objective_score):
        self.options  = options
        self.get_objective_score = get_objective_score
        geneinfo = []
        for i in range(initial_set):
            x = random.randint(0, 2 ** len(self.options))
            geneinfo.append(self.generate_conf(x))
        fitness = []
        self.begin = time.time()
        self.dep = []
        self.times = []
        for x in geneinfo:
            tmp = self.get_objective_score(x)
            fitness.append(-1.0 / tmp)
            
        self.pop = [(x, fitness[i]) for i, x in enumerate(geneinfo)]
        self.pop = sorted(self.pop, key=lambda x:x[1])
        self.best = self.selectBest(self.pop)
        self.dep.append(1.0/self.best[1])
        self.times.append(time.time() - self.begin)

    def generate_conf(self, x):
        comb = bin(x).replace('0b', '')
        comb = '0' * (len(self.options) - len(comb)) + comb
        conf = []
        for k, s in enumerate(comb):
            if s == '1':
                conf.append(1)
            else:
                conf.append(0)
        return conf

    def selectBest(self, pop):
        return pop[0]
        
    def selection(self, inds, k):
        s_inds = sorted(inds, key=lambda x:x[1])
        return s_inds[:int(k)]

    def crossoperate(self, offspring):
        dim = len(self.options)
        geninfo1 = offspring[0][0]
        geninfo2 = offspring[1][0]
        pos = random.randrange(1, dim)

        newoff = []
        for i in range(dim):
            if i>=pos:
                newoff.append(geninfo2[i])
            else:
                newoff.append(geninfo1[i])
        return newoff

    def mutation(self, crossoff):
        dim = len(self.options)
        pos = random.randrange(1, dim)
        crossoff[pos] = 1 - crossoff[pos]
        return crossoff

    def GA_main(self):
        for g in range(iters):
            selectpop = self.selection(self.pop, 0.5 * initial_set)
            nextoff = []
            while len(nextoff) != initial_set:
                offspring = [random.choice(selectpop) for i in range(2)]
                crossoff = self.crossoperate(offspring)
                muteoff = self.mutation(crossoff)
                fit_muteoff = self.get_objective_score(muteoff)
                nextoff.append((muteoff, -1.0 / fit_muteoff))
            self.pop = nextoff       
            self.pop = sorted(self.pop, key=lambda x:x[1])
            self.best = self.selectBest(self.pop)
            self.times.append(time.time() - self.begin)
            self.dep.append(1.0/self.best[1])

        return self.dep, self.times

if __name__ == '__main__':
    stats = []
    times = []
    parser = argparse.ArgumentParser(description="Args needed for BOCA tuning compiler.")
    parser.add_argument('--bin-path',
                        help='Specify path to compilation tools.',
                        metavar='<directory>', required=True)
    parser.add_argument('--driver',
                        help='Specify name of compiler-driver.',
                        metavar='<bin>', required=True)
    parser.add_argument('--linker',
                        help='Specify name of linker.',
                        metavar='<bin>', required=True)
    parser.add_argument('--libs',
                        help='Pass comma-separated <options> on to the compiler-driver.',
                        nargs='*', metavar='<options>', default='')
    parser.add_argument('-o', '--output',
                        help='Write output to <file>.',
                        default='a.out', metavar='<file>')
    parser.add_argument('-p', '--execute-params',
                        help='Pass comma-separated <options> on to the executable file.',
                        nargs='+', metavar='<options>')
    parser.add_argument('-src', '--src-dir',
                        help='Specify path to the source file.',
                        required=True, metavar='<directory>')
    args = parser.parse_args()

    if not os.path.exists(LOG_DIR):
        os.system('mkdir ' + LOG_DIR)

    make_params = {}
    boca_params = {}
    bin_path = args.bin_path
    if not bin_path.endswith(os.sep):
        make_params['bin_path'] = args.bin_path
    else:
        make_params['bin_path'] = args.bin_path[:-1]
    make_params['driver'] = args.driver
    make_params['linker'] = args.linker
    if args.libs:
        make_params['libs'] = args.libs
    make_params['output'] = args.output
    if args.execute_params:
        make_params['execute_params'] = args.execute_params
    make_params['src_dir'] = args.src_dir

    e = Executor(**make_params)
    space = {}
    stats = []
    times = []
    for i in range(begin2end):
        run = GA(e.o3_opts, e.get_objective_score)
        dep, ts = run.GA_main()
        print('middle result')
        print(dep)
        stats.append(dep)
        times.append(ts)

    vals = []
    for j, v_tmp in enumerate(stats):
        max_s = 0
        for i, v in enumerate(v_tmp):
            max_s = max(max_s, v)
            v_tmp[i] = max_s

    print(times)
    print(stats)

    for i in range(iters):
        tmp = []
        for j in range(begin2end):
            tmp.append(times[j][i])
        vals.append(np.mean(tmp))

    print(vals)

    vals = []
    for i in range(iters):
        tmp = []
        for j in range(begin2end):
            tmp.append(stats[j][i])
        vals.append(np.mean(tmp))

    print(vals)

    vals = []
    for i in range(iters):
        tmp = []
        for j in range(begin2end):
            tmp.append(stats[j][i])
        vals.append(np.std(tmp))

    print(vals)
