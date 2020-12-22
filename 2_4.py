""" This file is created as a template for question 2.4 in DD2434 - Assignment 2.

    We encourage you to keep the function templates as is.
    However this is not a "must" and you can code however you like.
    You can write helper functions however you want.

    You do not have to implement the code for finding a maximum spanning tree from scratch. We provided two different
    implementations of Kruskal's algorithm and modified them to return maximum spanning trees as well as the minimum
    spanning trees. However, it will be beneficial for you to try and implement it. You can also use another
    implementation of maximum spanning tree algorithm, just do not forget to reference the source (both in your code
    and in your report)! Previously, other students used NetworkX package to work with trees and graphs, keep in mind.

    We also provided an example regarding the Robinson-Foulds metric (see Phylogeny.py).

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py file),
    and modify them as needed. In addition to the sample files given to you, it is very important for you to test your
    algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format.

    Note that the sample files are tab delimited with binary values (0 or 1) in it.
    Each row corresponds to a different sample, ranging from 0, ..., N-1
    and each column corresponds to a vertex from 0, ..., V-1 where vertex 0 is the root.
    Example file format (with 5 samples and 4 nodes):
    1   0   1   0
    1   0   1   0
    1   0   0   0
    0   0   1   1
    0   0   1   1

    Also, I am aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). I wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    For this assignment, we gave you a single tree mixture (q2_4_tree_mixture).
    The mixture has 3 clusters, 5 nodes and 100 samples.
    We want you to run your EM algorithm and compare the real and inferred results
    in terms of Robinson-Foulds metric and the likelihoods.
    """
import numpy as np
import matplotlib.pyplot as plt
import sys

hash_q={}

def save_results(loglikelihood, topology_array, theta_array, filename):
    """ This function saves the log-likelihood vs iteration values,
        the final tree structure and theta array to corresponding numpy arrays. """

    likelihood_filename = filename + "_em_loglikelihood.npy"
    topology_array_filename = filename + "_em_topology.npy"
    theta_array_filename = filename + "_em_theta.npy"
    print("Saving log-likelihood to ", likelihood_filename, ", topology_array to: ", topology_array_filename,
          ", theta_array to: ", theta_array_filename, "...")
    np.save(likelihood_filename, loglikelihood)
    np.save(topology_array_filename, topology_array)
    np.save(theta_array_filename, theta_array)


def em_algorithm(seed_val, samples, num_clusters, max_num_iter=100):
    """
    This function is for the EM algorithm.
    :param seed_val: Seed value for reproducibility. Type: int
    :param samples: Observed x values. Type: numpy array. Dimensions: (num_samples, num_nodes)
    :param num_clusters: Number of clusters. Type: int
    :param max_num_iter: Maximum number of EM iterations. Type: int
    :return: loglikelihood: Array of log-likelihood of each EM iteration. Type: numpy array.
                Dimensions: (num_iterations, ) Note: num_iterations does not have to be equal to max_num_iter.
    :return: topology_list: A list of tree topologies. Type: numpy array. Dimensions: (num_clusters, num_nodes)
    :return: theta_list: A list of tree CPDs. Type: numpy array. Dimensions: (num_clusters, num_nodes, 2)

    This is a suggested template. Feel free to code however you want.
    """

    # Set the seed
    np.random.seed(seed_val)
    num_nodes=samples.shape[1]
    num_samples=samples.shape[0]
    loglikelihood = []
    topology_list = []
    theta_list = []
    # TODO: Implement EM algorithm here.
    from Tree import TreeMixture
    tm = TreeMixture(num_clusters=num_clusters, num_nodes=samples.shape[1])
    tm.simulate_pi(seed_val=seed_val)
    tm.simulate_trees(seed_val=seed_val)
    old_prob=-1
    new_prob=cal_probability(tm,samples)
    loglikelihood.append(new_prob)
    r=np.zeros([num_samples,num_clusters])
    iteration=0
    while abs(new_prob-old_prob)>10**-6 and iteration<max_num_iter:
        hash_q.clear()
        iteration+=1
        print('iteration: ',iteration)
        for n in range(num_samples):
            for k in range(num_clusters):
                r[n][k]=tm.pi[k]*calculate_likelihood(tm.clusters[k].get_topology_array(), tm.clusters[k].get_theta_array(), samples[n])/sum([tm.pi[i]*calculate_likelihood(tm.clusters[i].get_topology_array(), tm.clusters[i].get_theta_array(), samples[n]) for i in range(num_clusters)])
        tm.pi=[sum([r[n][k] for n in range(num_samples)])/num_samples for k in range(num_clusters)]
        tm.pi=tm.pi/sum(tm.pi)
        for k in range(num_clusters):
            from Kruskal_v1 import Graph
            G=Graph(num_nodes)
            for t in range(1,num_nodes):
                G.addEdge(0,t,I_q(r,k,samples,x_s=0,x_t=t))
            for s in range(1,num_nodes):
                for t in range(1,num_nodes):
                    if s != t:
                        G.addEdge(s,t,I_q(r,k,samples,x_s=s,x_t=t))
            t=G.maximum_spanning_tree()
            t_topo=np.array(get_topo(t))
            theta=[np.array(get_theta(r,k,samples,t_topo,0))] 
            for nodes in range(1,num_nodes):
                theta.append(np.array(get_theta(r,k,samples,t_topo,nodes)))
            from Tree import Tree
            tr=Tree()
            tr.load_tree_from_direct_arrays(t_topo,theta)
            tm.clusters[k]=tr
        old_prob=new_prob
        new_prob=cal_probability(tm,samples)
        loglikelihood.append(new_prob)
    for i in range(num_clusters):
        topology_list.append(tm.clusters[i].get_topology_array())
        theta_list.append(tm.clusters[i].get_theta_array())
    loglikelihood = np.array(loglikelihood)
    topology_list = np.array(topology_list)
    theta_list = np.array(theta_list)
    print(loglikelihood)
    return loglikelihood, topology_list, theta_list

def cal_probability(mix_tree,samples):
    prob=0
    for k in range(len(mix_tree.pi)):
        prob_k=0
        for sample in samples:
            prob_k+=calculate_likelihood(mix_tree.clusters[k].get_topology_array(),mix_tree.clusters[k].get_theta_array(),sample)
        prob_k=mix_tree.pi[k]*prob_k/len(samples)
        prob+=prob_k
    return np.log(prob+sys.float_info.min)

def calculate_likelihood(topo,theta,sample):
    result=1
    for i in range(len(sample)):
        p=1
        if i==0:
            p*=theta[0][sample[i]]
        else:
            current=i
            ancestor=int(topo[current])
            p*=theta[i][int(sample[i])][int(sample[ancestor])]
        result*=p
    return result


def cal_q(r,k,X,x_s=None,v_s=None,x_t=None,v_t=None):
    if (k,x_s,v_s,x_t,v_t) in hash_q:
        return hash_q[(k,x_s,v_s,x_t,v_t)]
    else:
        top=0
        for sample in range(len(X)):
            correct=True
            if x_s!=None and X[sample][int(x_s)]!=v_s:
                correct=False
            if x_t!=None and X[sample][int(x_t)]!=v_t:
                correct=False
            if correct:
                top+=r[sample][k]
        bottom=sum([r[i][k] for i in range(len(r))])
        result=top/bottom
        hash_q[(k,x_s,v_s,x_t,v_t)]=result
        return result

def get_topo(tree):
    num_node=len(tree)+1
    topo=np.zeros(num_node)
    used=np.zeros(num_node)
    topo[0]=np.nan
    used[0]=1
    while sum(used)!=num_node:
        for id in range(len(used)):
            if used[id]!=1:
                for edge in tree:
                    if edge[0]==id:
                        if used[edge[1]]==1:
                            topo[id]=edge[1]
                            used[id]=1
                    elif edge[1]==id:
                        if used[edge[0]]==1:
                            topo[id]=edge[0]
                            used[id]=1
    return topo

def get_theta(r,k,X,topo,node):
    if node==0:
        return [cal_q(r,k,X,x_s=0,v_s=a) for a in range(2)]
    else:
        result= [[cal_q(r,k,X,x_s=node,v_s=a,x_t=topo[node],v_t=b)/cal_q(r,k,X,x_t=topo[node],v_t=b) for b in range(2)] for a in range(2)]
        result=[result[i]/sum(result[i]) for i in range(2)]
        return result
    
def I_q(r,k,X,x_s=None,x_t=None):
    result=0
    for a in range(2):
        for b in range(2):
            if cal_q(r,k,X,x_s=x_s,v_s=a,x_t=x_t,v_t=b)==0:
                result+=0
            else:
                result+=cal_q(r,k,X,x_s=x_s,v_s=a,x_t=x_t,v_t=b)*np.log(cal_q(r,k,X,x_s=x_s,v_s=a,x_t=x_t,v_t=b)/(cal_q(r,k,X,x_s=x_s,v_s=a)*cal_q(r,k,X,x_t=x_t,v_t=b)))
    return result

def main():


    #orgin code,hide when you do 2.4.14
    
    seed_val2 = 123
    sample_filename = "data/q2_4/q2_4_tree_mixture.pkl_samples.txt"
    output_filename = "q2_4_results.txt"
    real_values_filename = "data/q2_4/q2_4_tree_mixture.pkl"
    num_clusters = 3
    

    #code for 2.4.14, hide them when you are running the first part of the question
    num_clusters=3
    num_nodes=5
    num_samples=100
    seed_val=400
    seed_val2=600
    from Tree import TreeMixture
    te = TreeMixture(num_clusters=num_clusters, num_nodes=num_nodes)
    te.simulate_pi(seed_val=seed_val)
    te.simulate_trees(seed_val)
    te.sample_mixtures(num_samples,seed_val=seed_val)
    filename="s="+str(num_samples)+"n="+str(num_nodes)+"c="+str(num_clusters)+".pkl"
    te.save_mixture(filename,save_arrays=True)
    output_filename=filename+"_result"
    real_values_filename = filename
    sample_filename=filename + "_samples.txt"
     #end code for 2.4.14, hide them when you are running the first part of the question




    samples = np.loadtxt(sample_filename, delimiter="\t", dtype=np.int32)
    num_samples, num_nodes = samples.shape
    print("\tnum_samples: ", num_samples, "\tnum_nodes: ", num_nodes)
    print("\tSamples: \n", samples)

    print("\n2. Run EM Algorithm.\n")

    loglikelihood, topology_array, theta_array = em_algorithm(seed_val2, samples, num_clusters=num_clusters)

    print("\n3. Save, print and plot the results.\n")

    save_results(loglikelihood, topology_array, theta_array, output_filename)

    for i in range(num_clusters):
        print("\n\tCluster: ", i)
        print("\tTopology: \t", topology_array[i])
        print("\tTheta: \t", theta_array[i])



    if real_values_filename != "":
        print("\n4. Retrieve real results and compare.\n")
        print("\tComparing the results with real values...")

        print("\t4.1. Make the Robinson-Foulds distance analysis.\n")
        # TODO: Do RF Comparison
        import dendropy
        from Tree import Tree
        tns = dendropy.TaxonNamespace()
        real=[]
        infer=[]
        for i in range(num_clusters):
            filename=real_values_filename+"_tree_"+str(i)+"_newick.txt"
            with open(filename, 'r') as input_file:
                newick_str = input_file.read()
            t = dendropy.Tree.get(data=newick_str, schema="newick", taxon_namespace=tns)
            print("\tTree "+str(i)+": ", t.as_string("newick"))
            real.append(t)

        print("\nLoad Inferred Trees")
        filename =output_filename+"_em_topology.npy" # This is the result you have.
        topology_list = np.load(filename)
        print(topology_list.shape)
        print(topology_list)

        for i in range(num_clusters):
            rt = Tree()
            rt.load_tree_from_direct_arrays(topology_list[0])
            rt = dendropy.Tree.get(data=rt.newick, schema="newick", taxon_namespace=tns)
            print("\tInferred Tree"+ str(i)+": ", rt.as_string("newick"))
            rt.print_plot()
            infer.append(rt)


        print("\n4.2 Compare trees and print Robinson-Foulds (RF) distance:\n")
        matrix=np.zeros([num_clusters,num_clusters])
        for i in range(num_clusters):
            for j in range(num_clusters):
                matrix[i][j]=dendropy.calculate.treecompare.symmetric_difference(real[i], infer[j])

        print(matrix)

        print("\n4.2. Make the likelihood comparison.\n")
        # TODO: Do Likelihood Comparison
    
    
    
    from Tree import TreeMixture
    tm_exact=TreeMixture(num_clusters=num_clusters, num_nodes=samples.shape[1])
    tm_exact.load_mixture(real_values_filename)
    exact_probability=cal_probability(tm_exact,samples)
    plt.figure(figsize=(8, 3))
    plt.title("sample = "+str(samples.shape[0])+"node = "+str(samples.shape[1])+"clusters = "+str(num_clusters))
    plt.subplot(121)
    plt.plot(np.exp(loglikelihood), label='Estimated')
    plt.plot(np.exp([exact_probability for _ in range(len(loglikelihood))]), label='Exact')
    plt.ylabel("Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.subplot(122)
    plt.plot(loglikelihood, label='Estimated')
    plt.plot([exact_probability for _ in range(len(loglikelihood))], label='Exact')
    plt.ylabel("Log-Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.legend(loc=(1.04, 0))
    plt.savefig('s='+str(num_samples)+'n='+str(num_nodes)+'c='+str(num_clusters))
    print('probability',cal_probability(tm_exact,samples))   
    plt.show()

if __name__ == "__main__":
    main()
