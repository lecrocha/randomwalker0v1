#=========================================================================================================================================================
# Author: Luis E C Rocha  - Ghent University, Belgium  - 26.09.2022
#
# Description: 	This file contains a simple implementation of the Random Walk model on a grid
#              	1. first install streamlit using "pip install streamlit" 
#               2. when you run streamlit, it will open a tab in your default browser with the streamlit application *it works as a webpage hosted at the following URL:  Local URL: http://localhost:8501
#
#=========================================================================================================================================================

# Import essential modules/libraries
import numpy as np
import pandas as pd
import random as rg
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from time import sleep  # this is used to add a delay for each time step

#===============================================================================================================
# THIS IS THE DEFINITION OF THE CLASS FOR THE MODEL
class RandomWalker:

    #===================================================================
    # This method initialises the system
    def __init__(self, N, prob_hop, boundary):

        # These are the parameters of the model, input by the user
        self.N = N
        self.prob_hop = prob_hop
        self.boundary = boundary

        # This is the size of the system - Ps: this little trick just makes sure the system (grid) will have integer length for a square grid
        NN = int(np.sqrt(self.N))**2
        self.grid_size = int(np.sqrt(NN))

        self.grid = np.zeros([self.grid_size, self.grid_size])

        #-----------------------------------------------------------------------
        # INITIAL POSITION OF THE RANDOM WALKER
        self.pos_x = rg.randint(0, self.grid_size-1)
        self.pos_y = rg.randint(0, self.grid_size-1)
        self.grid[self.pos_x, self.pos_y] = 1

    #====================================================================
    # MODEL: This method updates the position of the RW in the grid
    def run(self):

        # This flag will tell us if the RW got out of the grid - this is for the absorbing boundary conditions
        flag = 0
        
        # first test if the RW will hop to a neighbour cell
        # I just added this extra feature in the model. That means, before choosing a neighbour, you first check 
        # if the RW will hop or stay still at this current time
        # if a more advanced model, you can define this probability according to specific mechanisms you want to model
        # e.g. probability might depend on RW energy/resources/income/age, it might depend on past behaviour, etc
        p = rg.random()
        if p < self.prob_hop:

            # Choose a direction for the RW to hop
            next_position_direction = rg.randint(0, 3)

            # tip: To test your code, set:
            #next_position_direction = 1
            # Then, the RW will always move to the same direction and you know what to expect

            # Calculates the next position of the RW
            pos_xx, pos_yy = self.next_position(next_position_direction)
            
            #print(next_position_direction, pos_xx, pos_yy)	
            
            # if the returned position is negative (see method "next_position"), the RW went outside the grid.
            # In this case, it disappears and flag=1 so the time loop in the main code stops
            if (pos_xx == -1):

                self.grid[self.pos_x, pos_yy] = 0
                flag = 1

            elif (pos_yy == -1):

                self.grid[pos_xx, self.pos_y] = 0
                flag = 1

            # if the RW did not go out of the grid, then, update its position
            else:

                self.grid[pos_xx, pos_yy] = 1
                self.grid[self.pos_x, self.pos_y] = 0

                self.pos_x = pos_xx
                self.pos_y = pos_yy

        # returns the flag whether the RW disappeared or not
        return(flag)

    #====================================================================
    # This method calculates the next x,y position of the walker - considering different boundary conditions
    def next_position(self, direction):

        pos_xx = self.pos_x
        pos_yy = self.pos_y

        # I defined:
        # direction 0 as going to the left
        # direction 1 as going to the south
        # direction 2 as going to the right
        # direction 3 as going to the north

        # For periodic boundary conditions
        if self.boundary == "Periodic":
            if direction == 0:
                #if it goes to the left, but it's on position 0, then, move to the other side, following the periodic boundary conditions
                if self.pos_x == 0:
                    pos_xx = self.grid_size-1
                #otherwise, just move to the left
                else:
                    pos_xx = self.pos_x - 1

            elif direction == 1:
                if self.pos_y == 0:
                    pos_yy = self.grid_size - 1
                else:
                    pos_yy = self.pos_y - 1

            elif direction == 2:
                if self.pos_x == self.grid_size - 1:
                    pos_xx = 0
                else:
                    pos_xx = self.pos_x + 1

            elif direction == 3:
                if self.pos_y == self.grid_size - 1:
                    pos_yy = 0
                else:
                    pos_yy = self.pos_y + 1

        # For mirror boundary conditions
        elif self.boundary == "Mirror":
            if direction == 0:
                if self.pos_x == 0:
                    pos_xx = 1
                else:
                    pos_xx = self.pos_x - 1

            elif direction == 1:
                if self.pos_y == 0:
                    pos_yy = 1
                else:
                    pos_yy = self.pos_y - 1

            elif direction == 2:	
                if self.pos_x == self.grid_size-1:
                    pos_xx = self.grid_size-2
                else:
                    pos_xx = self.pos_x + 1

            elif direction == 3:
                if self.pos_y == self.grid_size-1:
                    pos_yy = self.grid_size-2
                else:
                    pos_yy = self.pos_y + 1

        # For absorbing boundary conditions
        elif self.boundary == "Absorbing":
            if direction == 0:
                if self.pos_x == 0:
                    pos_xx = -1
                else:
                    pos_xx = self.pos_x - 1

            elif direction == 1:
                if self.pos_y == 0:
                    pos_yy = -1
                else:
                    pos_yy = self.pos_y - 1

            elif direction == 2:	
                if self.pos_x == self.grid_size-1:
                    pos_xx = -1
                else:
                    pos_xx = self.pos_x + 1

            elif direction == 3:
                if self.pos_y == self.grid_size-1:
                    pos_yy = -1
                else:
                    pos_yy = self.pos_y + 1

        # direction 0 as going to the left
        # direction 1 as going to the south
        # direction 2 as going to the right
        # direction 3 as going to the north

        return(pos_xx, pos_yy)


#===============================================================================================================
# VISUALISATION OF THE MODEL DYNAMICS USING THE streamlit FRAMEWORK (see more on https://streamlit.io/)

#--------------------------------------------------------------------------------------------------
# Title of the visualisation - shows on screen

st.title("Random Walk model")

# This method is used to fix the random number generator. Use it for testing. Remove it to make analysis
#np.random.seed(0)

#--------------------------------------------------------------------------------------------------
# Methods to interactively collect input variables

N = st.sidebar.slider("Population Size", 5, 500, 100)
prob_hop = st.sidebar.slider("Prob Hopping", 0.0, 1.0, 0.5)
boundary = st.sidebar.radio("Boundary Conditions", ('Periodic', 'Mirror', 'Absorbing'))
no_iter = st.sidebar.number_input("Number of Iterations", 1)
speed = st.sidebar.number_input("Speed Simulation", 0.0, 1.0, 0.75)

#--------------------------------------------------------------------------------------------------
# Initialise the object   - Note that when one runs the code, the selected parameters will be passed here during the initialisation of the object via "self" method

randomwalker = RandomWalker(N, prob_hop, boundary)

#--------------------------------------------------------------------------------------------------
# Create placeholders for plot, iteration text, and progress bar
plot_placeholder = st.empty()

# This functions shows a progress bar
show_iteration = st.empty()
progress_bar = st.progress(0)

# Display the initial state
# lattice that will show the positions of the random walkers at the current time step
fig, ax = plt.subplots()
ax.axis('off')
cmap = ListedColormap(['white', 'red'])
# The grid is multiplied by 0 so the method only draws a grid without any values on it
ax.pcolormesh(0 * randomwalker.grid, cmap=cmap, edgecolors='k', linewidths=0.5)

# Attention: To understand the equivalence between our matrix here and how pcolormesh draws the matrix on the screen, 
#see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pcolormesh.html

# This function draws the figure (in object) fig on the screen
plot_placeholder.pyplot(fig)

# Close the figure instances
plt.close(fig)

# This functions shows a progress bar
show_iteration.text("Step 0")

#===============================================================================================================
# RUN THE DYNAMICS (IF THE USER CLICKS ON "Run") FOLLOWING THE RULES DEFINED ABOVE IN METHOD "RUN"

if st.sidebar.button('Run'):

    # Run the simulation for no_iter iterations, i.e. total time of the simulation
    # Repeat routines below for each time step i
    for i in range(no_iter):

        # Add a little time delay otherwise the patterns update too fast on the screen
        sleep(1.0-speed)

        # Call the method "Run" with the interaction rules
        # Here you can teste the other algorithms to update the state and then visualise what happens
        flag = randomwalker.run()
        
        #--------------------------------------------------------------------
        # Visualisation of the evolution

        # Create a new figure for the updated grid state
        fig, ax = plt.subplots()
        
        # lattice with the positions of all random walkers
        ax.axis('off')
        ax.pcolormesh(randomwalker.grid, cmap=cmap, edgecolors='k', linewidths=0.5)

        # Draws the figure (in the object fig) on the screen
        plot_placeholder.pyplot(fig)

        # Closes the figure instance (to replot them in the next time step)
        plt.close(fig)

        # Updates the progress bar
        show_iteration.text("Step %d" %(i+1))
        progress_bar.progress( (i+1.0)/no_iter )
        #------------------------------------------------------------------

        # if the RW got out of the grid, stop the simulation
        if flag:
            break
