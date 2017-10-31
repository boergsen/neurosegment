import numpy as np
import matplotlib.pyplot as plt
from core.methods import create_2d_edge_graph
import opengm
from core.data.calcium_imaging import JaneliaData
import IPython

# collect underpants
jd=JaneliaData(dummy_data=True)
sub = jd.subvolumes[0]
print sub.name
unary_potentials = sub.ssvm_pack_unary_features(['feats_pm_++'], normalize=True, bias=True)
pairwise_potentials = sub.ssvm_pack_pairwise_features(['feats_xcorr_green'], bias=True)
n_states = 2
n_nodes = len(unary_potentials)
edges = create_2d_edge_graph(512,512)
n_factors = n_nodes + edges.shape[0]

# set up graphical model in opengm
gm = opengm.gm(np.ones(n_nodes, dtype=opengm.label_type) * n_states)
gm.reserveFactors(n_factors)
gm.reserveFunctions(n_factors, 'explicit')
unaries = np.require(unary_potentials, dtype=opengm.value_type) * -1.0
fidUnaries = gm.addFunctions(unaries)
visUnaries = np.arange(n_nodes, dtype=opengm.label_type)
gm.addFactors(fidUnaries, visUnaries)
secondOrderFunctions = -np.require(pairwise_potentials, dtype=opengm.label_type)
fidSecondOrder = gm.addFunctions(secondOrderFunctions)
gm.addFactors(fidSecondOrder, edges.astype(np.uint64))

# do inference
inference= {
    #'inf_dd' : opengm.inference.DualDecompositionSubgradient,
    'inf_bp' : opengm.inference.BeliefPropagation,
    #'inf_trws' : opengm.inference.TrwsExternal,
    #'inf_tree_reweighted_bp' : opengm.inference.TreeReweightedBp,
    #'inf_gibbs' : opengm.inference.Gibbs,
    #'inf_lf' : opengm.inference.LazyFlipper,
    #'inf_icm' : opengm.inference.Icm,
    #'inf_dp' : opengm.inference.DynamicProgramming,
    #'inf_aef' : opengm.inference.AlphaExpansionFusion,
    'inf_gc' : opengm.inference.GraphCut,
    #'inf_loc' : opengm.inference.Loc,
    #'inf_mqpbo' : opengm.inference.Mqpbo,
    #'inf_ae' : opengm.inference.AlphaExpansion
}

fig, axes = plt.subplots(2, 2)
fig.set_size_inches(12,12)
axes = axes.ravel()
plt.ion()
fig_cnt = 0
for method in inference.iterkeys():
    print 'doing inference method', method
    inf = inference[method](gm)
    print 'i am okaz'
    res = inf.arg().astype(np.int)
    print 'still okay'
    # try:
    #     energy = gm.evaluate(res)
    # except:
    #     energy = 42
    axes[fig_cnt].imshow(res.reshape(512,512), cmap='gist_heat')
    axes[fig_cnt].set_title('%s (energy: %f)' % (method, 42))
    axes[fig_cnt].set_xticks([])
    axes[fig_cnt].set_yticks([])
    fig_cnt += 1

plt.savefig('out/only_ogm.png', dpi=300)
print 'Safe and sound.'

IPython.embed()
