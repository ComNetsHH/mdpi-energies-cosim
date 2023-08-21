import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import json
import argparse
import numpy as np
import math
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter

plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 36})
# plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rcParams['hatch.linewidth'] = 1.5
# plt.rcParams['markeredgewidth'] = 2

fig = plt.figure(figsize=(13, 8), dpi=100) # 13, 7
ax = fig.gca()
linewidth = 3
ax_list = []

def create_config_comp_palette(component_keys, configs, color_map):
    pal = {}
    for k in component_keys:
        for c in configs:
            pal_key = f'{k} ({c})'
            pal[pal_key] = color_map[component_keys.index(k)]
    return pal

palette = {
    'simDuration': 'tab:blue',
    'Total': 'tab:blue',
    'communication': 'tab:orange',
    'ICT': 'tab:orange',
    'communication (INET)': 'tab:orange',
    'communication (abstract)': 'brown',
    'step': 'tab:green',
    'FMU': 'tab:green',
    'Energy system': 'tab:green',
    'rpc': 'tab:red',
    'cell manager': 'tab:red',
    'Cell manager': 'tab:red',
    'setReal': 'tab:purple',
    'setString': 'tab:brown',
    'fmi2DoStep (exclusive)': 'lightgreen',
    'fmi2DoStep (inclusive)': 'darkgreen',
    'fmi2GetStringStatus (exclusive)': 'pink',
    'fmi2GetStringStatus (inclusive)': 'red'
}

palette_scalability = {
    'Total (111)': '#1f78b4',
    'Total (13)': '#a6cee3',
    'ICT (111)': '#ff7f00',
    'ICT (13)': '#fdbf6f',
    'Energy system (111)': '#33a02c',
    'Energy system (13)': '#b2df8a',
}

parser = argparse.ArgumentParser(description='Plotting execution time of different Co-Simulation parts.')
parser.add_argument('result_file', metavar='f', type=str, help='path to the file with exported execution times')
parser.add_argument('result_file_2', metavar='f2', type=str, help='path to the second file with exported execution times for combined plot', nargs='?')
parser.add_argument('-t', '--total', help='plot total execution time', action='store_true')
parser.add_argument('-c', '--communication', help='plot communication overhead time', action='store_true')
parser.add_argument('-fo', '--fmu-overhead', help='plot overhead of communication with FMU for variable delta t', action='store_true')
parser.add_argument('-a', '--all', help='plots execution time of all captured functions', action='store_true')
parser.add_argument('-dt', '--delta-t', help='compare step duration based on delta t', action='store_true')
parser.add_argument('-nlg', '--no-log', help='use normal, non-logarithmic y-axis scale', action='store_true')
parser.add_argument('-sc', '--scalability-comparison', help='compare 13- and 111-household networks in terms of execution times', action='store_true')
parser.add_argument('-s', '--save', help='name of the filename to save chart under', type=str)
parser.add_argument('-maxy', '--max-y', help='set the upper limit for y-axis', type=float)
parser.add_argument('-in', '--include', help='only process the results from specified config', type=str)
parser.add_argument('-stop', '--stop-at', help='to be used together with --active-powers flag to stop at the specified timestamp', type=int)
parser.add_argument('-ap', '--active-powers', help='Plot timeseries of total power consumption in the entire network', action='store_true')
parser.add_argument('-v', '--verbose', help='print plotted values', action='store_true')

args = parser.parse_args()
scale = 'log' if not args.no_log else 'linear'

def plot_clustered_stacked(dfall, labels=None, title="",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
    labels is a list of the names of the dataframe, used for the legend
    title is a string for the title of the plot
    H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_yscale('log')
    axe.set_ylabel('Execution time (s)')

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1])
    axe.add_artist(l1)
    plt.tight_layout()
    return axe


def process_profiler_capture(data):
    print(args.result_file)
    print('fmi2GetStringStatus: ', data[data['Name'].str.contains('fmi2GetStringStatus')]['Exclusive secs'].tolist().pop(), ' s (exclusive)')
    print('fmi2GetStringStatus: ', data[data['Name'].str.contains('fmi2GetStringStatus')]['Inclusive secs'].tolist().pop(), ' s (inclusive)')
    print('fmi2DoStep: ', data[data['Name'].str.contains('fmi2DoStep')]['Exclusive secs'].tolist().pop(), ' s (exclusive)')
    print('fmi2DoStep: ', data[data['Name'].str.contains('fmi2DoStep')]['Inclusive secs'].tolist().pop(), ' s (inclusive)')

def get_profile_cap_kind(row):
    return f"{row['Name']} ({row['Type'].lower()})"

def rename_communication(row):
    if 'communication' in row['Name']:
        return 'ICT'
    return row['Name']

def rename_cell_manager(row):
    if 'manager' in row['Name']:
        return 'Cell manager'
    return row['Name']

def rename_fmu_to_energy(row):
    if 'FMU' in row['Name']:
        return 'Energy system'
    return row['Name']

def rename_step_to_fmu(row):
    if 'step' in row['Name']:
        return 'FMU'
    return row['Name']

def rename_sim_duration(row):
    if 'simDuration' in row['Name']:
        return 'Total'
    return row['Name']

def find_first_el_index_less_than(mylist, value):
    for i in range(len(mylist)):
        if mylist[i] > value:
            return i-1
    return len(mylist)-1

def convert_json_to_df_timeseries(data):
    df_list = []

    def get_experiment_label(experiment_name):

        if 'Late' in experiment_name:
            return 'Communication outage'
        elif 'Prevented' in experiment_name:
            return 'Reliable communication'
        else:
            return 'No communication'        

    for run_key in data:
        row = data[run_key]

        for record in row['vectors']:
            tstop_index = find_first_el_index_less_than(record['time'], args.stop_at if args.stop_at else 99999)
            for i in range(0, tstop_index):
                df_list.append(
                    {
                        'Name': record['name'],
                        'Experiment': get_experiment_label(row['attributes']['experiment']),
                        'Time': record['time'][i],
                        'Value': record['value'][i]/1000 * (-1) # convert W to kW and change the sign
                    }
                )

    return pd.DataFrame(df_list)

def convert_json_to_df(data):
    """
    Converts result data exported from OMNeT++ as JSON to pandas dataframes
    """
    # print('converting json to df')
    df_list = []
    for run_key in data:
        row = data[run_key]
        dt = float(row["itervars"]["dt"])
        # print('delta t = ', dt)

        for i in range(0, len(row['scalars']), 2):
            stat_records = [row['scalars'][i], row['scalars'][i+1]]
            stat_name = stat_records[0]['name'].split(':')[0]
            mean_record = [r for r in stat_records if ':mean' in r['name']].pop()
            count_record = [r for r in stat_records if ':count' in r['name']].pop()

            # print(stat_name, ': mean = ', mean_record['value'], ', count = ', count_record['value'])

            if dt == 0.01:
                continue

            if args.include and row['attributes']['experiment'] != args.include:
                continue

            df_list.append(
                {
                    'Name': stat_name,
                    'Count': count_record['value'],
                    'Mean': mean_record['value'],
                    'Delta t': dt,
                    'Experiment': row['attributes']['experiment'],
                    'Measurement': row['attributes']['measurement'],
                    'Replication': row['attributes']['replication']
                }
            )

    return pd.DataFrame(df_list)

def plot_profile_capture():
    df_list = [
        {
            'Name': 'fmi2GetStringStatus',
            'Type': 'Inclusive',
            'Delta t': 0.01,
            'Value': 1894.84
        },
        {
            'Name': 'fmi2GetStringStatus',
            'Type': 'Exclusive',
            'Delta t': 0.01,
            'Value': 908.949
        },
        {
            'Name': 'fmi2DoStep',
            'Type': 'Inclusive',
            'Delta t': 0.01,
            'Value': 1087.69
        },
        {
            'Name': 'fmi2DoStep',
            'Type': 'Exclusive',
            'Delta t': 0.01,
            'Value': 0.0830312
        },
        {
            'Name': 'fmi2GetStringStatus',
            'Type': 'Inclusive',
            'Delta t': 1,
            'Value': 19.477
        },
        {
            'Name': 'fmi2GetStringStatus',
            'Type': 'Exclusive',
            'Delta t': 1,
            'Value': 9.25548
        },
        {
            'Name': 'fmi2DoStep',
            'Type': 'Inclusive',
            'Delta t': 1,
            'Value': 14.4136
        },
        {
            'Name': 'fmi2DoStep',
            'Type': 'Exclusive',
            'Delta t': 1,
            'Value': 0
        },
        {
            'Name': 'fmi2GetStringStatus',
            'Type': 'Inclusive',
            'Delta t': 3,
            'Value': 7.05907
        },
        {
            'Name': 'fmi2GetStringStatus',
            'Type': 'Exclusive',
            'Delta t': 3,
            'Value': 3.26714
        },
        {
            'Name': 'fmi2DoStep',
            'Type': 'Inclusive',
            'Delta t': 3,
            'Value': 6.06842
        },
        {
            'Name': 'fmi2DoStep',
            'Type': 'Exclusive',
            'Delta t': 3,
            'Value': 0
        },
        {
            'Name': 'fmi2GetStringStatus',
            'Type': 'Inclusive',
            'Delta t': 10,
            'Value': 4.89934
        },
        {
            'Name': 'fmi2GetStringStatus',
            'Type': 'Exclusive',
            'Delta t': 10,
            'Value': 2.17235
        },
        {
            'Name': 'fmi2DoStep',
            'Type': 'Inclusive',
            'Delta t': 10,
            'Value': 4.76307
        },
        {
            'Name': 'fmi2DoStep',
            'Type': 'Exclusive',
            'Delta t': 10,
            'Value': 0
        },
        {
            'Name': 'fmi2GetStringStatus',
            'Type': 'Inclusive',
            'Delta t': 100,
            'Value': 3.77043
        },
        {
            'Name': 'fmi2GetStringStatus',
            'Type': 'Exclusive',
            'Delta t': 100,
            'Value': 1.56197
        },
        {
            'Name': 'fmi2DoStep',
            'Type': 'Inclusive',
            'Delta t': 100,
            'Value': 3.83937
        },
        {
            'Name': 'fmi2DoStep',
            'Type': 'Exclusive',
            'Delta t': 100,
            'Value': 0
        },
        {
            'Name': 'fmi2GetStringStatus',
            'Type': 'Inclusive',
            'Delta t': 1000,
            'Value': 3.70525
        },
        {
            'Name': 'fmi2GetStringStatus',
            'Type': 'Exclusive',
            'Delta t': 1000,
            'Value': 1.59592
        },
        {
            'Name': 'fmi2DoStep',
            'Type': 'Inclusive',
            'Delta t': 1000,
            'Value': 3.74879
        },
        {
            'Name': 'fmi2DoStep',
            'Type': 'Exclusive',
            'Delta t': 1000,
            'Value': 0
        },
    ]

    df = pd.DataFrame(df_list)
    df['Kind'] = df.apply(lambda row: get_profile_cap_kind(row), axis=1)

    df_step = df[ df['Name'] == 'fmi2DoStep' ]
    df_string = df[ df['Name'] == 'fmi2GetStringStatus' ]

    pivot_df = df.pivot(index='Delta t', columns='Kind', values='Value')

    pivot_df_step = df_step.pivot(index='Delta t', columns='Type', values='Value')
    pivot_df_string = df_string.pivot(index='Delta t', columns='Type', values='Value')
    
    # pivot_df_step.plot(kind = 'bar', stacked = True, ax=ax)
    # pivot_df_string.plot(kind = 'bar', stacked = True, ax=ax)
    

    # sns.barplot(data=df, x='Delta t', y='Value', hue='Kind', palette=palette, ax=ax)

    plot_clustered_stacked( [pivot_df_step, pivot_df_string], ['fmi2DoStep', 'fmi2GetStringStatus'] )

    # sns.barplot(data=df[ df['Name'] == 'fmi2DoStep' ], x='Delta t', y='Value', hue='Kind', palette=palette, ax=ax)
    # sns.barplot(data=df[ df['Name'] == 'fmi2GetStringStatus' ], x='Delta t', y='Value', hue='Kind', palette=palette, ax=ax)
    # ax.set_yscale('log')
    # ax.set_ylabel('Execution time (s)')
    # ax.legend(title=None)

def calc_total_exec_time(row):
    return int(row['Count']) * float(row['Mean'])

def rename_rpc_type(row):
    if row['Name'] == 'rpc':
        return 'cell manager'
    else:
        return row['Name']

def derive_kind_configs_comparison(row):
    """Derives the kind label for comparison of 3 configs: abstract communication, INET, INET with Optimal Power Flow"""
    stat = row["Name"] # name of the statistic, e.g. simDuration, communication, etc.
    experiment = row['Experiment']

    if 'Abstract' in experiment:
        experiment = 'AC'
    elif 'Cellmanager' in experiment:
        experiment = 'INET + OPF'
    else:
        experiment = 'INET'

    return f'{convert_state_name_v2(stat)} ({experiment})'

def convert_state_name_v2(old_name):
    if 'FMU' in old_name:
        return 'Energy system'
    elif 'simDuration' in old_name:
        return 'Total'
    else:
        return 'ICT'

def derive_kind(row):
    experiment = 'INET' if 'INET' in str(row['Experiment']) else 'Abstract'
    num_households = int(re.search('_(\d+)_HH', row['Experiment']).group(1))    

    return f'{convert_state_name_v2(row["Name"])} ({num_households})'

def strip_vec_name(row):
    return row['Name'].replace('ExecTime:vector', '')

def calc_communication_time(df):
    print('Calculating communication time, experiments: ', df['Experiment'].unique().tolist())

    rows_to_append = []

    for rep, rgroup in df.groupby('Replication'):
        for e, egroup in rgroup.groupby('Experiment'):
            # print('e : ', e, ' df experiments: ', df['Experiment'])
            for m, group in egroup.groupby('Measurement'):
                print(f'{e} {m} {rep}')

                sim_dur = float(group[group['Name'] == 'simDuration']['Total'].tolist().pop())

                time_spent_other_ops = group[group['Name'] != 'simDuration']['Total'].tolist()
                time_spent_other_ops = [0 if math.isnan(i) else i for i in time_spent_other_ops]

                # print('Sim duration: ', sim_dur)
                # print('List of other durations: ', group[group['Name'] != 'simDuration']['Total'].tolist(), 'total: ', sum(group[group['Name'] != 'simDuration']['Total'].tolist())) 

                time_spent_communication = sim_dur - sum(time_spent_other_ops)

                # print(f'dt = {group["Delta t"].tolist()[0]}, experiment: ', e, ', replication = {} time spent communicating: ', time_spent_communication)
                
                dt = int(group["Delta t"].tolist()[0])

                # if dt == 3 and e == 'INET_13_HH_Cellmanager':
                #     print(f'dt = {dt}, experiment: {e}, replication = {rep}, time spent communicating: {time_spent_communication}')

                rows_to_append.append(
                    {
                        "Experiment": e,
                        "Measurement": m,
                        "Name": "communication",
                        "Mean": time_spent_communication,
                        "Replication": rep,
                        "Count": 1,
                        "Total": time_spent_communication,
                        "Delta t": group['Delta t'].tolist().pop()
                    }
                )

    for row in rows_to_append:
        # print('appending row: ', row)
        df = df.append(row, ignore_index=True)
        
    return df

if 'capture' in args.result_file:
    simdata = pd.read_csv(args.result_file)
    # process_profiler_capture(simdata)
    plot_profile_capture()
    plt.xlabel(r'$\Delta_t$ (s)')
    plt.grid(axis='y')
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save)
    # plt.legend()
    plt.show()
    exit()
else:

    if 'json' in args.result_file:
        simdata = json.load(open(args.result_file))

        if args.active_powers:
            simdata = convert_json_to_df_timeseries(simdata)

            fig = plt.figure(figsize=(16, 9))
            ax = fig.gca()
            
            pal = {
                'No communication': 'tab:orange',
                'Reliable communication': 'tab:green',
                'Communication outage': 'tab:blue'
            }
            
            # sns.lineplot(data=simdata[simdata['Experiment'] != 'Reliable communication'], x='Time', y='Value', hue='Experiment', linewidth=3, ls='--', ax=ax, palette=pal)
            # sns.lineplot(data=simdata[simdata['Experiment'] == 'Reliable communication'], x='Time', y='Value', hue='Experiment', linewidth=3, ls='--', ax=ax, palette=pal)

            markers=['+', 'o', '*']
            mark_id = 0
            for exp, group in simdata.groupby('Experiment'):
                
                if exp == 'Reliable communication':
                    sns.lineplot(data=group, x='Time', y='Value', linewidth=13, ax=ax,
                                label=exp, color=pal[exp], ls=':')
                elif exp == 'No communication':
                    sns.lineplot(data=group, x='Time', y='Value', linewidth=6, label=exp, color=pal[exp])
                else:
                    sns.lineplot(data=group, x='Time', y='Value', linewidth=5, label=exp, color=pal[exp])

                mark_id += 1
                # sns.lineplot(x=traffic_rates, y=delays_reversed, label=r'expected (this work)', ax=ax, markersize=25, marker='o', linewidth=4.0, linestyle='--', color='#153AFF', markeredgecolor="#153AFF", markerfacecolor='none', markeredgewidth=3.0)


            # sns.lineplot(data=simdata, x='Time', y='Value', hue='Experiment', linewidth=3, ls='--', ax=ax, palette=pal, markers=['x', 'o', '*'])

            ax.axhline(y=200, color='red', ls='--', linewidth=linewidth)
            plt.ylabel('Total power consumption [kW]')
            plt.xlabel('Simulation time [s]')
            # plt.legend(title=None)
            leg = plt.legend()
            leg.remove()
            plt.grid()
            plt.tight_layout()
            plt.ylim([150, 240])
            
            if args.save:
                plt.savefig(args.save)
            else:
                plt.show()

            exit()
        else:
            simdata = convert_json_to_df(simdata)

        if args.result_file_2:
            simdata_2 = json.load(open(args.result_file_2))
            simdata_2 = convert_json_to_df(simdata_2)
            f, axs = plt.subplots(1, 2, sharey=False, sharex=True, figsize=(21, 9))
            ax_list = list(axs)

    else:
        simdata = pd.read_csv(args.result_file, sep='\t')
        simdata['Delta t'] = [float(re.search('dt=(\d+(\.\d+)?)', x).group(1)) for x in simdata['Measurement'].tolist()]
    
    simdata['Mean'] = [float(str(x).replace(',','')) for x in simdata['Mean'].tolist()]
    if args.result_file_2:
        simdata_2['Mean'] = [float(str(x).replace(',','')) for x in simdata_2['Mean'].tolist()]
        
    # print('experiments after: ', simdata['Experiment'].tolist())

if not args.all:

    delta_ts = simdata['Delta t'].unique().tolist()
    delta_ts.sort()

    if args.delta_t:
        simdata = simdata[simdata['Name'].str.contains('step')]

    if args.total:
        simdata['Total'] = simdata.apply(lambda row: calc_total_exec_time(row), axis=1)

        if args.fmu_overhead:
            simdata = simdata[simdata['Name'].str.contains('step')]
            # simdata = simdata[simdata['Experiment'].str.contains('Abstract')]

            df_list = []
            
            steps_total = simdata['Count'].unique().tolist()
            steps_total.sort(reverse=True)

            print('dts: ', delta_ts)
            for exp, egroup in simdata.groupby('Experiment'):
                total_fmu_simtime = np.mean(egroup[egroup['Delta t'] == 1000]['Total'].tolist())

                print('total FMU simtime: ', total_fmu_simtime)

                for rep, group in egroup.groupby('Replication'):
                    total_exec_times = group['Total'].unique().tolist()
                    total_exec_times.sort(reverse=True) # TODO: check if this is strictly necessary

                    print(f'#{rep} total exec times: ', total_exec_times)
                    
                    for i in range(len(delta_ts)):
                        if delta_ts[i] >= 1000:
                            continue
                        df_list.append(
                            {
                                'Experiment': 'INET' if 'INET' in exp else 'Abstract',
                                'Delta t': delta_ts[i],
                                'Replication': rep,
                                'FMU step time': (total_exec_times[i] - total_fmu_simtime) / steps_total[i] * 1000
                            }
                        )

                # total_exec_times = [(total_exec_times[i] - total_fmu_simtime) / steps_total[i] for i in range(len(total_exec_times))]


            # fig2 = plt.figure()
            # ax2 = fig2.gca()
                        
            # print('number of steps: ', steps_total)
            # print('communication with FMU: ', total_exec_times)

            # sns.barplot(x=delta_ts, y=total_exec_times)
            df_fmu_overhead = pd.DataFrame(df_list)
            df_fmu_overhead.rename(columns={"Experiment": "Communication"}, inplace=True)
            sns.barplot(data=df_fmu_overhead, x='Delta t', y='FMU step time', hue='Communication')
            plt.grid(axis='y')
            # bar2 = sns.barplot(x=delta_ts, y=[total_fmu_simtime for _ in delta_ts], label='FMU simtime')
            # for i, thisbar in enumerate(bar2.patches):
            #     # Set a different hatch for each bar
            #     if i < len(delta_ts):
            #         thisbar.set_hatch('x')
            # ax2.set(xlabel=r"$\Delta_t$", ylabel="Execution time in s (total)")
        else:
            sns.barplot(data=simdata, x='Delta t', y='Total', ax=ax)

        if args.verbose:
            for dt, group in simdata.groupby('Delta t'):
                print(f'dt = {dt}, mean = ', group['Total'].mean())

        plt.ylabel('Execution time [ms]')
        
        if args.fmu_overhead:
            delta_ts.pop()
            delta_ts.pop()

        ax.set_xticklabels([ fr'${i}$' for i in delta_ts])
        ax.set_xlabel(r'$\Delta t$ [s]')
        # plt.ylabel('Communication with FMU cost (s)')
        lgd = plt.legend(title='Communication', fontsize=28, title_fontsize=28)
        plt.yscale(scale)
        plt.yticks([0, 1, 2, 3, 4, 5, 6, 7])
        # plt.yticks([0.001, 0.01, 0.1, 1, 10, 100, 1400])
        # plt.ylim([0, 1400])
        if args.max_y:
            plt.ylim([0, args.max_y])

    else:

        if args.verbose:
            previous = 0
            for dt, group in simdata.groupby('Delta t'):
                print(f'dt = {dt}, mean = ', group['Mean'].mean(), end="")
                if previous != 0:
                    print(f' {round(group["Mean"].mean() / previous, 3)} times more')
                else:
                    print("")
                previous = group["Mean"].mean()

        sns.barplot(data=simdata, x='Delta t', y='Mean', ax=ax)
        plt.ylabel('Execution time in s (single step)')
        plt.yscale(scale)
        ax.set_xticklabels([ fr'${i}$' for i in delta_ts])
        # plt.yticks( [pow(10, i) for i in range(-3, 1)] )
        # plt.yticks([0.001, 0.01, 0.1, 1, 10, 100, 1400])
        # plt.ylim([0, 1400])
        if args.max_y:
            plt.ylim([0, args.max_y])

# Plot total execution times
else:

    simdata['Name'] = simdata.apply(lambda row: row['Name'].replace('ExecTime', '').replace(':vector', ''), axis=1)
    if args.result_file_2:
        simdata_2['Name'] = simdata_2.apply(lambda row: row['Name'].replace('ExecTime', '').replace(':vector', ''), axis=1)
    
    try:
        simdata.drop(columns=['Module'], inplace=True)
    except:
         None

    if args.total:
        simdata['Total'] = simdata.apply(lambda row: calc_total_exec_time(row), axis=1)
        simdata['Name'] = simdata.apply(lambda row: rename_rpc_type(row), axis=1)
        simdata = calc_communication_time(simdata)

        print('Num communication records: ', len(simdata[simdata['Name'].str.contains('communication')]['Total'].tolist()))

        simdata = simdata.sort_values(by=['Total'], ascending=False)

        if args.verbose:
            print(simdata['Experiment'].unique().tolist()[0])
            for dt, group in simdata.groupby('Delta t'):
                comm_exec_time = group[group['Name'].str.contains('communication')]['Total'].mean()
                fmu_exec_time = group[group['Name'].str.contains('step')]['Total'].mean()
                cm_exec_time = group[group['Name'].str.contains('cell')]['Total'].mean()
                total_exec_time = group[group['Name'].str.contains('simDuration')]['Total'].mean()

                print(f'dt = {dt}, fmu = {fmu_exec_time}, communication = {comm_exec_time}, cell manager = {cm_exec_time}, total = {total_exec_time}')

        if args.result_file_2:
            simdata_2['Total'] = simdata_2.apply(lambda row: calc_total_exec_time(row), axis=1)
            simdata_2['Name'] = simdata_2.apply(lambda row: rename_rpc_type(row), axis=1)
            simdata_2 = calc_communication_time(simdata_2)
            simdata_2 = simdata_2.sort_values(by=['Total'], ascending=False)

        if args.result_file_2:    
            print(simdata_2['Experiment'].unique().tolist()[0])
            for dt, group in simdata_2.groupby('Delta t'):
                comm_exec_time = group[group['Name'].str.contains('communication')]['Total'].mean()
                fmu_exec_time = group[group['Name'].str.contains('step')]['Total'].mean()

                print(f'dt = {dt}, fmu = {fmu_exec_time}, communication = {comm_exec_time}')

        if args.communication:
            # simdata['Name'] = simdata.apply(lambda row: calc_total_exec_time(row), axis=1)
            sns.barplot(data=simdata[simdata['Name'].str.contains('communication')], x='Delta t', y='Total', hue='Experiment', ax=ax)
        elif args.scalability_comparison:
            simdata['Name'] = simdata.apply(lambda row: rename_step_to_fmu(row), axis=1)
            simdata_2['Name'] = simdata_2.apply(lambda row: rename_step_to_fmu(row), axis=1)    
            # simdata = simdata[simdata['Experiment'].str.contains('Abstract')]
            # simdata = simdata[simdata['Experiment'].str.contains('INET')]
            # simdata = simdata[simdata['Delta t'] > 0.1]
        
            if not args.result_file_2:
                data_filtered = simdata[simdata['Name'].str.contains('simDuration') | simdata['Name'].str.contains('communication') | simdata['Name'].str.contains('FMU')]
                data_filtered['Kind'] = data_filtered.apply(lambda row: derive_kind(row), axis=1)
                sns.barplot(data=data_filtered, x='Delta t', y='Total', ax=ax, hue='Kind', palette=palette_scalability, hue_order=list(palette_scalability.keys())) 
            
            else:
                data_filtered = simdata[simdata['Name'].str.contains('simDuration') | simdata['Name'].str.contains('communication') | simdata['Name'].str.contains('FMU')]
                # data_filtered['Name'] = data_filtered.apply(lambda row: rename_sim_duration(row), axis=1)
                # data_filtered['Name'] = data_filtered.apply(lambda row: rename_communication(row), axis=1)
                data_filtered['Kind'] = data_filtered.apply(lambda row: derive_kind(row), axis=1)

                data_filtered_2 = simdata_2[simdata_2['Name'].str.contains('simDuration') | simdata_2['Name'].str.contains('communication') | simdata_2['Name'].str.contains('FMU')]
                data_filtered_2['Kind'] = data_filtered_2.apply(lambda row: derive_kind(row), axis=1)

                fig_new = plt.figure(figsize=(16, 7))
                ax_new = fig_new.gca()
                ax_new.set_xlabel(None)
                ax_new.set_xticks([])
                ax_new.set_ylabel(r'Execution time total (s)')
                ax_new.tick_params('x', labelrotation=30)
                
                box = ax_new.get_position()
                # ax_new.set_position([box.x0, box.y0,
                #     box.width, box.height * 0.7])
                
                # sns.barplot(data=data_filtered, x='Delta t', y='Total', ax=ax_list[0], hue='Kind', palette=palette_ bility, hue_order=list(palette_scalability.keys()))
                print('Kinds - ', data_filtered['Kind'].unique().tolist())
                bar1 = sns.barplot(data=data_filtered[data_filtered['Delta t'] == 3], x='Delta t', y='Total', hue='Kind', ax=ax_new, palette=palette_scalability, hue_order=list(palette_scalability.keys()))
                

                paddings = [10, 25, 10, 30, 10, 10]
                for i, c in enumerate(bar1.containers):
                    bar1.bar_label(c,fmt='%.1f', padding=paddings[i])
                
                # for rect in list(bar1):
                #     height = rect.get_height()
                #     plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom')

                # sns.barplot(data=data_filtered_2, x='Delta t', y='Total', ax=ax_list[1], hue='Kind', palette=palette_scalability, hue_order=list(palette_scalability.keys()))

        else:
            # sns.barplot(data=simdata[simdata['Delta t'] > 0.01], x='Delta t', y='Total', hue='Name', palette=palette, ax=ax)
            
            simdata['Name'] = simdata.apply(lambda row: rename_step_to_fmu(row), axis=1)

            print('Names: ', simdata['Name'].unique().tolist())

            num_experiments = len(simdata['Experiment'].unique().tolist())
            if num_experiments == 1:

                simdata = simdata[simdata['Name'].str.contains('simDuration') | simdata['Name'].str.contains('communication') | simdata['Name'].str.contains('FMU') | simdata['Name'].str.contains('manager')]

                simdata['Name'] = simdata.apply(lambda row: rename_fmu_to_energy(row), axis=1)
                simdata['Name'] = simdata.apply(lambda row: rename_sim_duration(row), axis=1)
                simdata['Name'] = simdata.apply(lambda row: rename_communication(row), axis=1)
                simdata['Name'] = simdata.apply(lambda row: rename_cell_manager(row), axis=1)

                sns.barplot(data=simdata, x='Delta t', y='Total', hue='Name', palette=palette, ax=ax)
                ax.set_xticklabels([ fr'${i}$' for i in [0.1, 1, 3, 10, 100, 1000]])
            else:
                f, axs = plt.subplots(1, num_experiments, sharey=True, sharex=True, figsize=(21 if num_experiments > 2 else 16, 9))
                ax_list = list(axs)
                print(f'detected {num_experiments} experiments, subplots: ', len(ax_list))
                
                ax_id = 0

                simdata = simdata[simdata['Name'].str.contains('simDuration') | simdata['Name'].str.contains('communication') | simdata['Name'].str.contains('FMU')]
                simdata = simdata[simdata['Delta t'] == 3]

                #### Grouping different configs into one plot for a single delta t
                # TODO: derive directly from dataframe
                pal = create_config_comp_palette(['Total', 'ICT', 'Energy system'], ['INET + OPF', 'INET', 'AC'], ['tab:blue', 'tab:orange', 'tab:green'])
                
                simdata['Kind'] = simdata.apply(lambda row: derive_kind_configs_comparison(row), axis=1)

                fig_new = plt.figure(figsize=(18, 9), dpi=600)
                ax_new = fig_new.gca()
                ax_new.set_xlabel(None)
                ax_new.set_xticks([])
                ax_new.set_ylabel(r'Execution time (s)')
                ax_new.tick_params('x', labelrotation=30)
                bar = sns.barplot(data=simdata, x='Delta t', y='Total', hue='Kind', ax=ax_new, palette=pal, hue_order=list(pal.keys())) # palette=pal, ) 

                hatches = ['-', '+', 'x', '\\', '*', 'O']

                for i, thisbar in enumerate(bar.patches):
                    # Set a different hatch for each bar
                    # if (i + 1) % 2 == 0:
                    #     thisbar.set_hatch('-')
                    if i == 0 or i % 3 == 0:
                        thisbar.set_hatch('|')
                    
                    if (i + 1) % 3 == 0:
                        thisbar.set_hatch('X')

                    if (i + 1) == 2 or (i + 1) == 5 or (i + 1) == 8:
                        thisbar.set_hatch('O')

                for c, i in enumerate(bar.containers):
                    if c >= 1 and c < 6:
                        if c == 4:
                            bar.bar_label(i,fmt='%.1f',padding=35)
                        elif c == 5:
                            bar.bar_label(i,fmt='%.1f',padding=50)
                        elif c == 3:
                            bar.bar_label(i,fmt='%.1f',padding=15)
                        else:
                            bar.bar_label(i,fmt='%.1f',padding=25)
                    else:
                        bar.bar_label(i,fmt='%.1f')

                # for e, group in simdata.groupby('Experiment'):
                #     sns.barplot(data=group, x='Delta t', y='Total', hue='Name', palette=palette, ax=ax_list[ax_id])
                #     figtitle = ''

                #     if 'Abstract' in e:
                #         figtitle = 'Abstract Communication'
                #     elif 'Cellmanager' in e:
                #         figtitle = 'INET + Optimal Power Flow'
                #     else:
                #         figtitle = 'INET'

                #     ax_list[ax_id].set_title(figtitle, fontsize=32)
                #     # ax_list[ax_id].set_yticks([pow(10, i) for i in range(-5, 2)])
                #     ax_id += 1
            
        # sns.barplot(data=simdata, x='Delta t', y='Total', hue='Name', palette=palette, ax=ax)

        if not args.result_file_2:
            plt.ylabel('Execution time in s (total)')
        
        if args.max_y:
            plt.ylim([0, args.max_y])
        
    else:
        sns.barplot(data=simdata, x='Delta t', y='Mean', hue='Name', palette=palette, ax=ax)
        plt.ylabel('Execution time in s (single instance)')

    plt.yscale(scale)

    if args.max_y:
        plt.ylim([0, args.max_y])
    plt.legend(title='')


# if args.result_file_2 or len(simdata['Experiment'].unique().tolist()) > 1 and not args.fmu_overhead:
#     handles, labels = ax_list[0].get_legend_handles_labels()
#     f.legend(handles, labels, loc='upper center', ncols=2   , bbox_to_anchor=(0.5, 1), fontsize=32)
#     for axi in ax_list:
#         axi.grid(axis='y')    
#         axi.get_legend().remove()
#         axi.set_xlabel(r'$\Delta t$')
#         axi.tick_params('x', labelrotation=30)
#         delta_ts = simdata['Delta t'].unique().tolist()
#         delta_ts.sort()

#         axi.set_xticklabels([ fr'${i}$' for i in [0.1, 1, 3]])
#         axi.set_yscale('log')
        
#         # Only for 13 vs 111 household comparison
#         # axi.set_yticks([pow(10, i) for i in [-1, 0, 1, 2, 3]])
#         axi.set_yticks([pow(10, i) for i in [-5, -4, -2, -1, 0, 1, 2]])
#         # axi.set_ylim([None, 1800])

#         if axi == ax_list[0]:
#             axi.set_ylabel(r'Execution time in s (total)')
#         else:
#             axi.set_ylabel(None)

#     plt.tight_layout(pad=1.5)
#     plt.subplots_adjust(top=0.7)
    
#     # Comment out for 13 vs 111 household comparison
#     # if not args.no_log:
#     #     plt.yticks([pow(10, i) for i in [-5, -4, -2, 0, 1, 2, 3]])
    
#     # plt.xticks(rotation=30)
# else:
    plt.xlabel(None)

    if "INET_13_HH_Cellmanager" in args.include:
        plt.xlabel(r'$\Delta t$ [s]')
        plt.yticks([pow(10, i) for i in [-1, 0, 1, 2]])
    else:
        plt.yticks([pow(10, i) for i in [0, 1, 2, 2.3]])
        plt.xticks([])
    # plt.yticks([pow(10, i) for i in [-1, 0, 1, 2]])
    # plt.yticks([pow(10, i) for i in [0, 1, 2]])
    # plt.ylim([None, pow(10, 2)])
    plt.ylabel('Execution time [s]')
    plt.grid(axis='y')
    # lgd = plt.legend(loc='upper center', ncols=2, bbox_to_anchor=(0.5, 1.3), fontsize=28)
    lgd = plt.legend(loc='upper center', ncols=3, bbox_to_anchor=(0.5, 1.4), fontsize=28)
# plt.subplots_adjust(top=0.7)
# plt.tight_layout()


if args.save:
    plt.savefig(args.save, bbox_extra_artists=(lgd,), bbox_inches='tight')
# plt.legend()
plt.show()