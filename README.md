# My AI course projects
Some AI course projects. Reference to some online and undergraduate course resources.

# 1. Edited Burnt Pancake Problem
A program that receives an order of 4 bottom-burnt pancakes and prints the solution that BFS and A* search will find for going from the Start state to the Goal (ordered pancakes and all burnt-side down).

- Each of the pancakes has an ID number that is consistent with their size followed by a letter “w” or “b”
- The largest pancake has an ID of 4, the next largest 3, the next 2, and the smallest has an ID of 1.
- The letter “w” refers to the unburnt side is up, and “b” shows that the burnt side is up. 
- The goal is to reach “1w2w3w4w”.

## Input
- The input should consist of pairs of four digits and one character, a hyphen, and one last character (#C#C#C#C-X)
- \# is ID of pancake, the first \# is the ID of pancake on the top
- C is character "w" or "b"
- X is character "b" or "a"; "b" is Breadth First Search , "a" is A* Search 

## Output
- For each state (except the final state), use the character “|” to show where the flip to go to the next step happens.
- For A*, also print the value for the actual cost (function g) and the value of the heuristic function (function h) in each step

## Example
Input: 1w2b3w4b-a <br>
Output:<br>
1w2b|3w4b g:0,h:0 <br>
2w|1b3w4b g:2, h:2 <br>
2b1b3w4b| g:3, h:2 <br>
4w|3b1w2w g:7, h:4 <br>
4b3b1w2w| g:8, h:4 <br>
2b1b|3w4w g:12, h:2 <br>
1w2w3w4w g:14, h:0 <br>


Input: 1w2b3w4b-b <br>
Output: <br>
1w2b|3w4b <br>
2w|1b3w4b <br>
2b1b|3w4b <br>
1w2w3w4b| <br>
4w|3b2b1b <br>
4b3b2b1b| <br>
1w2w3w4w <br>


## Tie Breaking
When there is a tie between two nodes (same priority), replace “w” with 1 and “b” with 0 to obtain an eight-digit number. After that pick the node with a larger numerical ID chosen.

- For instance, if there is a tie between 4b3w2b1b and 3w4w2b1b, then 4b3w2b1b will be chosen as 40312010 > 31412010.





# 2. Alpha-Beta Pruning (Search)

Program should print the index of the terminal states that will be pruned using the alpha-beta search algorithm. The indexes are fixed and are shown in the figure below (0 to 11). 4 layer tree. Root node is MIN node.

Example
Input: <br>
2 4 13 11 1 3 3 7 3 3 2 2<br>
Output:<br>
3 6 7 10 11



Input: <br>
1 4 26 87 3 72 32 2 <br>
Output: <br>
10 11 <br>




# 3. Q-learning with 4x4 Board
- Board is 4x4, i.e. 16 squares, each of which have unique ID 
'''
13 14 15 16
9  10 11 12
5  6  7  8
1  2  3  4
'''
- There are five special squares on the board. Start, goal(2), forbidden, and wall squares
- The remaining 11 squares are empty and ordinary squares. 
- The starting square is fixed and always at square 2.
-  The location of the two goals, forbidden, and wall squares are determined from the input.
-  In this problem the living reward for every action (each step) is r = -0.1. The discount rate is γ = 0.1, and the learning rate is α = 0.3
-  The reward for performing the exit action (the only available action) in both goal squares is +100, and for the forbidden square is -100. The agent cannot enter or pass through the wall square. After hitting the wall, the agent’s position will not be updated. It will remain in the same square and will keep getting a -0.1 reward every time it hits the wall.
-  For the purpose of exploring the board, use an ε-greedy method with ε = 0.5. This means that with the probability ε, the agent acts randomly, and with the probability 1-ε, it acts on current policy. 
-  In order to have a similar random value use 1 as the seed value of your random function.
-  Can set up convergence as a maximum number of iterations to 100,000. But not necessary according to my own experience.

## Input
4 numbers, one character, possibly an additional number 
- The first four numbers show the location of the two goals, forbidden, and wall squares respectively
- The fourth item is either character “p” or “q”
  - "p" refers to printing the optimal policy Phi
  - "q" refers to the optimal Q-values, if it’s “q”, there will be an additional number at the end
- Assume the five special squares are distinct (non-overlapping).

## Output 
If the input contains “p”, program has to print the best action that should be chosen for each square or in other words print Phi*. To do this, in separate lines print each state’s index and the action.

## Example
Input:<br>
15 12 8 6 p<br>
Output:<br>
1 up <br> 2 right<br> 3 up <br>4 left<br> 5 up<br> 6 wall-square <br> 7 up<br>
8 forbid<br>
9 up<br>
10 up <br>11 up<br> 12 goal<br> 13 right <br>14 right <br>15 goal<br> 16 up<br>


# 4. Perceptron and Logistic Regression

Note: this implementation is different from the normal expressions that you can see from Machine Learning courses. I was confused why my code migrated from a machine learning project failed at the tests. Because although in different logics, they're still the same thing. <br>

## Perceptron
The first function receives n (<100) triplets of x1, x2, y from the user, and returns the value for w (weight) that the perceptron algorithm computes to classify the inputs. Here x1 and x2 show the input features for each sample, and y shows the class. Consider a binary classification with two possible values for y: -1 and +1. Use the same format for input and output as the example below. Note that before the actual input, another input character P will determine the call for the perceptron function.


Example Input: <br>
P (0, 2,+1) (2, 0, -1) (0, 4,+1) (4, 0, -1) \#P indicates perceptron <br>

Example output: <br>
-2.0, 0.0 # referring to w=[-2.0, 0.0]

Use the same procedure for updating w, as discussed in UCB's  slides (𝑤 = 𝑤 + 𝑦*. 𝑓). Start from w=[0,0], and update w by a maximum of n*100 times, where n is the number of input samples (100 times iterating over all of the input samples).

## Logistic Regression
The second function receives a similar input as described above with the only difference in the first input character (L instead of P). The desired task will still be binary classification. The output, in this case, will be printing the probability values that logistic regression computes for each input belonging to the positive class. Set alpha (learning rate) equal to 0.1.<br>

Example Input: <br>
L (0, 2,+1) (2, 0, -1) (0, 4, -1) (4, 0, +1) (0, 6, -1) (6, 0, +1) \#L indicates Logistic Reg. <br>

Example output: <br>
0.29 0.71 0.14 0.86 0.06 0.94 <br>

For the logistic regression use the basic procedure introduced in notes originally written by Nikhil Sharma, Pieter Abbeel and Dan Klein. 
(𝑤 = 𝑤 + α * ∇𝑔(𝑤)). Start from w=[0,0], and update w by a maximum of n*100 times, where n is the number of input samples (100 times iterating over all of the input samples). Note that when g(z)=sigmoid(z), we will have g'(z)=g(z)(1-g(z)). Not necessarily needed here, but for this case, you can further simplify this calculation (at the end of the [doc](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/pdfs/40%20LogisticRegression.pdf)).

