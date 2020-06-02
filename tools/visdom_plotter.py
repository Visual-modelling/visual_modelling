import visdom
import numpy as np
import getpass
import torch
import sixfour, os 
from subprocess import PIPE, run

# TO DO:
# -Plot successfully
# -Have each keep track of highest accuracy so far


#So it begins
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        if getpass.getuser() == "jumperkables":
            self.viz = visdom.Visdom()
        if getpass.getuser() == "crhf63":
            self.viz = visdom.Visdom(server="jumperkables@jerry.clients.dur.ac.uk", port=8097)
        self.env = env_name
        # Remove an enviroment if it exists
        if env_name in self.viz.get_env_list():
            self.viz.delete_env(env_name)

        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
    def text_plot(self, var_name, t):
        if var_name not in self.plots:
            self.plots[var_name]    = self.viz.text(text=t, env=self.env)
        else:
            self.viz.text(text=t, env=self.env, win=self.plots[var_name])
    def im_plot(self, var_name, im):
        if var_name not in self.plots:
            self.plots[var_name]    = self.viz.image(img=im, env=self.env)
        else:
            self.viz.image(img=im, env=self.env, win=self.plots[var_name])
    
    def gif_plot(self, var_name, gif_path):
        gif_base64 = out('sixfour -i %s' % gif_path)
        html = '<html><head></head><body>'
        html += '<img src="data:image/gif;base64,{0}">'.format(gif_base64)
        html += '</body></html>'
        
        if var_name not in self.plots:
            self.plots[var_name]    = self.viz.text(text=html, env=self.env)
        else:
            self.viz.text(text=html, env=self.env, win=self.plots[var_name])


def out(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result.stdout

if __name__ == "__main__":
    plotter = VisdomLinePlotter("gif_test")
    aa = out('sixfour -i ~/d3_MSE_10-1.gif')   
    html = '<html><head></head><body>'
    html += '<img src="data:image/gif;base64,{0}">'.format(aa)
    html += '</body></html>'


