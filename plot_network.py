# This module contains functions for plotting for the network visualizer tool

from pandas import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community
import matplotlib.colorbar as cb
import seaborn as sns
from matplotlib import gridspec


def plot_network(Gtest, border_cols, md_network,
                 focal_node_name, edge_thresh, network_algo, map_degree, plot_border_col, draw_shortest_paths,
                 coexpression, colocalization, other, physical_interactions, predicted_interactions, shared_protein_domain):

    nodes = Gtest.nodes()
    node_series = Series(nodes)

    focal_node = list(node_series[node_series==focal_node_name].index)[0]

    numnodes = len(Gtest)

    # deal with case where no border_cols input
    if len(border_cols) < numnodes:
        border_cols = Series(np.ones(numnodes))
        border_cols.index = nodes

    # create a new dataframe mapping network_group value to edges
    e1e2 = Series(zip(list(md_network['Entity 1']), list(md_network['Entity 2']), list(md_network['Weight'])))
    e1e2NG_df = DataFrame(data={'e1e2': e1e2, 'Network_group': md_network['Network_group']})

    # find all coexpression edges
    sum(md_network['Network_group'] == 'Co-expression')


    # select 'Co-expression' edges
    edge_list_CE = list(e1e2NG_df.e1e2[e1e2NG_df.Network_group == 'Co-expression'])
    # select 'Co-localization' edges
    edge_list_CL = list(e1e2NG_df.e1e2[e1e2NG_df.Network_group == 'Co-localization'])
    # select 'Other' edges
    edge_list_OTHER = list(e1e2NG_df.e1e2[e1e2NG_df.Network_group == 'Other'])
    # select 'Physical interactions' edges
    edge_list_PI = list(e1e2NG_df.e1e2[e1e2NG_df.Network_group == 'Physical interactions'])
    # select 'Predicted' edges
    edge_list_PRED = list(e1e2NG_df.e1e2[e1e2NG_df.Network_group == 'Predicted'])
    # select 'Shared protein domain' edges
    edge_list_SPD = list(e1e2NG_df.e1e2[e1e2NG_df.Network_group == 'Shared protein domains'])

    edge_list_total = []
    if coexpression:
        edge_list_total.extend(edge_list_CE)
    if colocalization:
        edge_list_total.extend(edge_list_CL)
    if other:
        edge_list_total.extend(edge_list_OTHER)
    if physical_interactions:
        edge_list_total.extend(edge_list_PI)
    if predicted_interactions:
        edge_list_total.extend(edge_list_PRED)
    if shared_protein_domain:
        edge_list_total.extend(edge_list_SPD)

    # find edges < threshold
    elarge = [(u, v) for (u, v, d) in edge_list_total if d > edge_thresh]
    esmall = [(u, v) for (u, v, d) in edge_list_total if d <= edge_thresh]

    # create temp network for calculations
    Gtemp = nx.Graph()
    Gtemp.add_nodes_from(Gtest.nodes())
    Gtemp.add_edges_from(elarge)
    
    if network_algo == 'community':
        fig = plt.figure(figsize=(20,25))
        gs = gridspec.GridSpec(2,1,height_ratios=[2,1])
        ax = plt.subplot(gs[0])
    else:
        fig,ax=plt.subplots(figsize=(20, 15))
    pos_2pi = nx.pygraphviz_layout(Gtest,prog='twopi', root=focal_node_name)
    #pos_2pi = nx.graphviz_layout(Gtemp, prog='twopi', root=focal_node_name, args='')

    if network_algo == 'spl':

        all_SPs = nx.single_source_shortest_path_length(Gtemp, focal_node_name)
        cols = Series(index=nodes)
        all_SPs = Series(all_SPs)
        #cols[all_SPs.index]=all_SPs.values
        cols[border_cols.index]=border_cols

        pos = pos_2pi  # tree layout
        cmap = 'RdYlGn'  # inverted cool colormap
        plot_star = 'on'  # plot focal star

        vmin = np.min(cols); vmax = np.max(cols)  # anchor colormap

        cbar_label = 'fold change'  # label for colorbar

    elif network_algo == 'clustering_coefficient':
        all_CCs = nx.cluster.clustering(Gtemp, nodes=nodes)
        all_CCs = Series(all_CCs)
        cols = Series(index=nodes)
        cols[all_CCs.index] = all_CCs.values

        pos = pos_2pi  # tree layout
        cmap = 'cool_r'
        plot_star = 'on'  # plot focal star

        vmin = None; vmax = None  # don't anchor colormap

        cbar_label = 'clustering coefficient'  # label for colorbar


    elif network_algo == 'hotnet2':
        F = heat_diffusion_matrix(Gtemp, .001)
        h = np.zeros(numnodes)  # initialize heat vector
        h[focal_node] = 1
        E = heat_update(F, h)
        cols = E[:, focal_node]
        cols[focal_node] = 0  # set focal node to 0 for better cmap scale
        cols[focal_node] = max(cols)
        cols = Series(cols)

        cols.index = nodes

        pos = pos_2pi  # tree layout
        cmap = 'cool'
        plot_star = 'on'  # plot focal star

        vmin = np.min(cols); vmax = np.max(cols)  # anchor colormap

        cbar_label = 'node heat'  # label for colorbar

    elif network_algo == 'pagerank':
        pr_G = nx.pagerank(Gtemp)
        cols = Series(index=nodes)
        pr_G = Series(pr_G)
        cols[pr_G.index]=pr_G.values

        pos = nx.spring_layout(Gtemp)  #nx.spring_layout(Gtemp,k=.8)  # spring layout
        cmap = 'cool'
        plot_star = 'on'  # plot focal star

        vmin = min(cols); vmax = max(cols)  # anchor colormap

        cbar_label = 'page rank'  # label for colorbar

    elif network_algo == 'community':
        partition = community.best_partition(Gtemp)

        partition = Series(partition)

        # calculate average foldchange in each community
        avg_FC_comm = border_cols[nodes].groupby(partition[nodes]).median()
        std_FC_comm = border_cols[nodes].groupby(partition[nodes]).std()
        FC_value_counts = partition.value_counts()
        FC_df = DataFrame({'avg community FC':avg_FC_comm,'std community FC':std_FC_comm,'value counts':FC_value_counts})

        # find community of focal gene
        focal_comm = partition[focal_node_name]

        num_comms = len(partition.unique())
        vmin = 0; vmax = num_comms-1  # anchor colormap

        cols = Series(index=nodes)
        cols[nodes] = list(partition[nodes])

        pos = nx.spring_layout(Gtemp)  #nx.spring_layout(Gtemp,k=.8)
        cmap = 'hsv' #'gist_rainbow'
        plot_star = 'on'  # plot focal star

        cbar_label = 'community ID'  # label for colorbar

    # first draw border nodes
    if plot_border_col:
        nodes_col = nx.draw_networkx_nodes(Gtemp,pos=pos,node_color=border_cols,node_size=1600,alpha=.7,cmap='RdYlGn')
        cbar = fig.colorbar(nodes_col,shrink=.5)
        cbar.set_label('fold change', size=18)


    degree = Gtemp.degree()
    degree = Series(degree)
    if map_degree:
        deg_max = np.max(degree)
        degree = degree/deg_max
        degree = degree*1000
        deg_norm = degree[nodes]
    else:
        deg_norm = degree*0+1000
    not_nan_nodes = list(border_cols[~np.isnan(border_cols)].index)

    # then draw main nodes
    nodes_col = nx.draw_networkx_nodes(Gtemp,node_color=cols[not_nan_nodes],pos=pos,node_size=deg_norm[not_nan_nodes],
                                       cmap=cmap,edgelist=[],with_labels=True,labels=nodes,font_size=9,alpha=.9,
                                       font_color='k',vmin=vmin, vmax = vmax,nodelist = not_nan_nodes)
    # draw unmeasured nodes
    nan_nodes = list(border_cols[np.isnan(border_cols)].index);
    nx.draw_networkx_nodes(Gtemp,node_color = 'w',pos=pos,node_size=deg_norm[nan_nodes],
                         nodelist=nan_nodes)

    nx.draw_networkx_labels(Gtemp,pos=pos,font_size=9,font_color='k')
    xtemp = []
    ytemp = []
    for i in nodes:
        xtemp.append(pos[i][0])
        ytemp.append(pos[i][1])

    nx.draw_networkx_edges(Gtemp,pos=pos,alpha=0.05,width=2)

    if draw_shortest_paths==True:
        # highligh shortest paths from focal node
        paths = nx.all_pairs_shortest_path(Gtemp)
        path_focal = paths[focal_node_name]

        edge_list = []
        for n in path_focal.keys():
            path_temp = path_focal[n]
            edge_temp = zip(path_temp,path_temp[1:])
            edge_list.extend(edge_temp)

        nx.draw_networkx_edges(Gtemp,edgelist=edge_list,pos=pos,alpha=.2,width=4,edge_color='deepskyblue')
    else:
    	plot_star = 'off'


    if plot_star == 'on':   # only plot central star if plot_star flag is on
        if np.isnan(cols[focal_node_name]):
            col_temp = 'w'
        else:
            col_temp = cols[focal_node_name]
        nx.draw_networkx_nodes(Gtemp,pos=pos,nodelist=[focal_node_name],node_color = col_temp,
                              node_size=4000,cmap=cmap,vmin=vmin,vmax=vmax,node_shape='*')

    plt.xlim(-3,3)
    plt.axis('equal')

    if network_algo=='community':
        numcoms = len(partition.unique())
        cbar = fig.colorbar(nodes_col,shrink=.5,ticks=range(numcoms))

        # calculate modularity and a text box
        mod = community.modularity(dict(partition),Gtemp)
        #plt.text(-1,-1.1,'graph modularity = %.2f' % mod, fontsize=16)
        plt.title('graph modularity = %.2f' % mod, fontsize=16)
    else:
        cbar = fig.colorbar(nodes_col,shrink=.5)
    cbar.set_label(cbar_label, size=18)

    plt.grid('off')
    ax.set_axis_bgcolor('white')
    plt.xticks([]); plt.yticks([])

    if network_algo=='community':
        ax1 = plt.subplot(gs[1])
        df_partition_FC = DataFrame({'partition':partition,'fold_change':border_cols});
        df_partition_FC = df_partition_FC.sort(columns='partition')

        palette_temp = sns.color_palette('hls',numcoms)
        sns.boxplot(x='partition',y='fold_change',data=df_partition_FC,palette = palette_temp,saturation=.9)

        # plot location of focal datapoint if plot_star
        if plot_star == 'on':
        	ax1.plot(focal_comm,avg_FC_comm[focal_comm],marker = '*',
                    	markersize=24,markerfacecolor='w',markeredgecolor='k')

        plt.xlabel('community id',fontsize=14)
        plt.ylabel('average fold change in group', fontsize=14)

    return fig


def heat_diffusion_matrix(G, beta):
    '''
    This function calculates the head diffusion matrix F (influence node j has on node i)
    F only needs to be calculated once per network, because it is independent of heat vector h
    inputs graph G, and insulation parameter beta
    '''

    numnodes = len(G)
    nodes = G.nodes()
    # loop over nodes
    F = np.zeros([numnodes, numnodes])
    W = np.zeros([numnodes, numnodes])
    for i in range(numnodes):
        for j in range(numnodes):
            if i != j:  # don't calc for i == j
                if G.has_edge(nodes[i], nodes[j]):  # check for edge existence
                    W[i, j] = 1.0/G.degree(nodes[j])  # normalized degree
                else:
                    W[i, j] = 0
    # diffusion matrix F: Fij represents influence g_j has on g_i
    F = beta*np.linalg.inv((np.identity(numnodes)-(1-beta)*W))

    return F

def heat_update(F, h):
    '''
    - This function updates E and h (exchanged heat matrix and heat vector) for a given beta (proxy for 1/t)
    - h is the initial 'heat' of each node
    --> Output (E) is basically the heat that will propagate to neighbor nodesin a certain time t.
        Heat will be more 'smoothed' for a longer t (smaller beta)
    NOTE: make sure this calculation is right...
    '''

    # make h a diagonal matrix
    hmat = np.diag(h)
    E = np.dot(F, hmat)  # np.dot for matrix multiplication

    return E