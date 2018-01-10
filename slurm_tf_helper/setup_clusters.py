#*** License Agreement ***
#
#High Energy Physics Deep Learning Convolutional Neural Network Benchmark (HEPCNNB) Copyright (c) 2017, The Regents of the University of California, 
#through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.
#
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#(1) Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#(2) Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer 
#in the documentation and/or other materials provided with the distribution.
#(3) Neither the name of the University of California, Lawrence Berkeley National Laboratory, U.S. Dept. of Energy nor the names 
#of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, 
#BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
#COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
#LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
#LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
#EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#You are under no obligation whatsoever to provide any bug fixes, patches, or upgrades to the features, 
#functionality or performance of the source code ("Enhancements") to anyone; however, 
#if you choose to make your Enhancements available either publicly, or directly to Lawrence Berkeley National Laboratory, 
#without imposing a separate written license agreement for such Enhancements, then you hereby grant the following license: a non-exclusive, 
#royalty-free perpetual license to install, use, modify, prepare derivative works, incorporate into other computer software, 
#distribute, and sublicense such enhancements or derivative works thereof, in binary and source code form.
#---------------------------------------------------------------

import tensorflow as tf

import sys

import os


def setup_slurm_cluster(num_ps=1):
    all_nodes = get_all_nodes()

    port = get_allowed_port()
    
    hostlist = [ ("%s:%i" % (node, port)) for node in all_nodes ] 
    ps_hosts, worker_hosts = get_parameter_server_and_worker_hosts(hostlist, num_ps=num_ps)

    proc_id, num_procs = get_slurm_proc_variables()
    
    num_tasks = num_procs - num_ps
    
    job_name = get_job_name(proc_id, num_ps)
    
    task_index = get_task_index(proc_id, job_name, num_ps)
    
    cluster_spec = make_cluster_spec(worker_hosts, ps_hosts)

    server = make_server(cluster_spec, job_name, task_index)
    
    return cluster_spec, server, task_index, num_tasks, job_name
    

def make_server(cluster_spec, job_name, task_index):
    server = tf.train.Server(cluster_spec,
                           job_name=job_name,
                           task_index=task_index)
    return server
    
    
def make_cluster_spec(worker_hosts, ps_hosts):
    if ps_hosts:
        cluster_spec = tf.train.ClusterSpec({
            "worker": worker_hosts,
            "ps": ps_hosts})
    else:
         cluster_spec = tf.train.ClusterSpec({
            "worker": worker_hosts})
    return cluster_spec
    
    
def get_task_index(proc_id, job_name, num_ps):
    
    if job_name == "ps":
        task_index = proc_id
    elif job_name == "worker":
        #expects a task_index for workers that starts at 0
        task_index = proc_id - num_ps
    return task_index
    
    
def get_slurm_proc_variables():
    proc_id  = int( os.environ['SLURM_PROCID'] )
    num_procs     = int( os.environ['SLURM_NPROCS'] )
    return proc_id, num_procs
    
def get_job_name(proc_id, num_ps):
    if proc_id < num_ps:
        job_name = "ps"
    else:
        job_name = "worker"
    return job_name
    
    
def get_parameter_server_and_worker_hosts(hostlist, num_ps=1):
    """assumes num_ps nodes used for parameter server (one ps per node)
    and rest of nodes used for workers"""
    ps_hosts = hostlist[:num_ps]
    worker_hosts = hostlist[num_ps:]
    return ps_hosts, worker_hosts
    
def get_allowed_port():
    allowed_port = 22222
    return allowed_port


def get_all_nodes():
    nodelist=expand_nodelist( os.environ['SLURM_NODELIST'])
    return nodelist
    

def expand_nodelist(node_string):
    if '[' in node_string:
        pref, suff  = node_string.split('[')

        suff = suff. split(']')[0].split(',')
        nodes =[]
        for s in suff:
            if '-' not in s:
                nodes.append("%s%s" % (pref, s))
                continue
            beg,end = s.split('-')
            num_len=len(beg)
            for id in range(int(beg),int(end) + 1):
                j= "%s%0" + str(num_len) + "d"
                nodes.append(j % (pref, id))
    else:
        nodes=[node_string]

    return nodes

if __name__ == "__main__":
    cluster, server, task_index, num_tasks, job_name = setup_slurm_cluster(num_ps=1)
