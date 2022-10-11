# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 09:42:58 2022

@author: stand
"""

# AI for Self Driving Car
# Importing the libraries


import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable



# Creating the architecture of the Neural Network
# Classe de base Init et Forward pour retourner les valeurs
# classe dérivée


class Network(nn.Module):
    # input_size=5 car 3 signaux, 2 orientations + ou -
    # nb_actions: avancer, droite ou gauche    
    def __init__(self, input_size=5, nb_action=3):
        # variables des parents via super
        super(Network, self).__init__()
        # variables locales stockées via self
        self.input_size = input_size
        self.nb_action = nb_action
        # Couche entrée Neurones et 30 neurones cachées
        self.fc1 = nn.Linear(input_size, 30)
        # Model simple 1couche entree, Une couche cachée puis sortie
        self.fc2 = nn.Linear(30, nb_action)
    
    # Fonction de propagation en avant. Renvoie les Q_Values
    def forward(self, state):
        # X sont les données entrée. Que signifie State?
        # Que signifie Q_Values?
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values




# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        # memoire entiere (~100.000)
        # nombre de transitions à retenir

    #ajoute les evennements dans la memoire
    #remplit la memoire en s'assurant le la capacity
    # event est le nouvel evennement qui va dans la memoire
    def push(self,event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    
    #Ajoute des échantillons dans la memoire
    def sample(self, batch_size):
        # permet de selectionner les transitions
        # zip combine en couples car on a deux listes
        # on va aligner les deux listes event par event
        # liste de memoires la position de l'historique et l'event
        samples= zip(*random.sample(self.memory, batch_size))
        #alignement tous ensemble sur le 1ere dimension grace au map
        return map(lambda x: Variable(torch.cat(x, dim=0)), samples)
        # torch.cat va concatener
        # Variable aide au calcul du gradient des events




# Implementing Deep QLearnings

class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        # input size taille couche d'entrée
        # combien d'entrees : 5
        #last_signal avec liste de taille avec les signaux 3 signaux et 2 var d'orinetatiuon
        # n_actions (left, straight, right) -20, 0, +20
        # gamma facteur reduction Belman = 0.9 vient de Equation Bellman.
        self.model = Network(input_size, nb_action)
        self.gamma = gamma
        self.reward_window = []
        self.memory = ReplayMemory(10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr= 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        # Unsqueeze permet de mettre dans un Batch.
        self.last_action = 0
        self.last_reward = 0

    
    def select_action(self, state):
        # SoftMax Fonction activation. Renvoie des probabilités.
        # On va chercher à Exacerber ces sorties. Avec un parametre Temperature.
        probs = F.softmax(self.model(state)*7, dim=1) # Ici T=7
        # Probabilités sortent de SoftMax. Aide model à choisir la meilleur probabilité
        # Multinomial sélectionne la plus grande des valeurs favorisée.
        # Pourquoi favoriser? 
        action = probs.multinomial(num_samples=1)           
        # Selectionne l'action effectuée.
        return action.data[0, 0]

# Entrainement par les Poids. Phase apprentissage est indépdendante de sélection Actions
# Sélection de Batch de Transition D'action enchainement pour entrainer le réseau de neurone.
# Comparaison des QCibles avec QValues. Calcul de fonction de Cout. Erreur.
# Retropropagation de Erreur. Mise à jour des poids.

    def learn(self, batch_state, batch_next_state, batch_action, batch_reward):
        # Batch est Groupe. Groupe de Transition est : Environnement à Etat initial. Environnement à Etat Final. Recompense. Action.
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        # Redimension des sorties.
        # Calcul des Cibles avec Equation de Bellman. Sorties doivent correspondre a état antérieur
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        # Creation de la sortie suivante. Etat t+1.
        # Detach permet de garder les références.
        # (1) action meilleur résultat. Vecteur Indice [0]
        targets = batch_reward + self.gamma * next_outputs
        # Correspond a Equation de Bellman R+Gamma*Liste actions possibles
        td_loss = F.smooth_l1_loss(outputs, targets)
        # Passage dans la fonction activation
        # Calcul du Cout. Il faut retropropager l'erreur.
        # On utilise PyTorch 
        self.optimizer.zero_grad()
        # Retropropagation. Algorithme de Gradient method zero_grad.
        td_loss.backward()
        # Application de Retropropagation sur vecteur de sortie.
        # Fonction de cout contient Backward.
        self.optimizer.step()
        # Optimizer Permet mise à Jour des Poids.


    def update(self, reward, new_signal):
    #metre a jour notre environnement et les variables subjacentes
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), self.last_reward))
        action = self.select_action(new_state)
                
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_action, batch_reward)

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        
        return action



    def score(self):
        # mesure d'avancement et score ok
        return sum(self.reward_window) / (len(self.reward_window) +1.)
        # reward_window contient les 1000 dernières recompenses. On fait la moyenne.



    def save(self):
        # sauvegarder dans un file les details
        # Methode state_dict permet de récuperer les poids du model.
        torch.save({"state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                    }, "last_brain.pth"
                   )


    def load(self):    
        # recuprer les params depuis save
        if os.path.isfile("last_brain.pth"):
            # OS permet d'explorer les fichiers
            print("=> Loading CheckPoint...")
            checkpoint = torch.load("last_brain.pth")
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            # Methode load_state_dict permet de charger l'objet précédent
        else:
            print("désolé, pas de fichier")
        
                
        
        
        
        
        
        