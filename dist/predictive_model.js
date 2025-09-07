/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   predictive_model.js                                :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: yohan <yohan@student.42.fr>                +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/09/07 10:00:50 by yohan             #+#    #+#             */
/*   Updated: 2025/09/07 12:21:24 by yohan            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

export const actions = [
    'up',
    'down',
    'left',
    'right',
    'up-right',
    'up-left',
    'down-right',
    'down-left',
    'none',
];

// is acc OP, like it's not even funny. I'll need to add noise or reduce paddle speed to allow losses
export class predictive_model {
    //______________________________________________________________________________//
    // mathematical helpers //
    //______________________________________________________________________________//
    dotProduct(a, b) {
        if (a.length !== b.length) {
            throw new Error("Vectors must have the same length for dot product");
        }
        let sum = 0;
        for (let i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    state_to_vector(state) {
        return [
            state.X_pos, state.Y_pos, state.Z_pos,
            state.Vx, state.Vy, state.Vz,
            state.X_paddle, state.Y_paddle,
            state.paddle_speed, state.paddle_height,
            state.time_to_wall_x, state.time_to_wall_y,
            state.x_dist_to_paddle, state.y_dist_to_paddle
        ];
    }
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    softmax(logits) {
        const maxLogit = Math.max(...logits); // for numerical stability
        const expScores = logits.map(v => Math.exp(v - maxLogit));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        return expScores.map(v => v / sumExp);
    }
    //______________________________________________________________________________//
    // learning //
    //______________________________________________________________________________//
    constructor(learning_rate = 0.1, weights) {
        this.num_inputs = 14;
        this.num_of_hidden_neurons = 12;
        this.num_outputs = 9; //possible outcomes
        this.best_W_hidden_input = [];
        this.best_W_hidden_output = [];
        this.best_bias_hidden_layer = [];
        this.best_bias_output_layer = [];
        this.bestScore = -Infinity;
        //weights:
        this.W_hidden_input = []; // hidden_neurons * num_inputs
        this.W_hidden_output = []; // num_outputs * hidden_neurons
        this.bias_hidden_layer = [];
        this.bias_output_layer = []; //allows neurons to function even when all weights are 0
        this.learning_rate = learning_rate;
        this.n_iter = 100;
        this.W_hidden_input = weights?.W_hidden_input ??
            Array(this.num_of_hidden_neurons).fill(0).map(() => Array(this.num_inputs).fill(0).map(() => Math.random() - 0.5));
        this.W_hidden_output = weights?.W_hidden_output ??
            Array(this.num_outputs).fill(0).map(() => Array(this.num_of_hidden_neurons).fill(0).map(() => Math.random() - 0.5));
        this.bias_hidden_layer = weights?.bias_hidden_layer ??
            Array(this.num_of_hidden_neurons).fill(0).map(() => Math.random() - 0.5);
        this.bias_output_layer = weights?.bias_output_layer ??
            Array(this.num_outputs).fill(0).map(() => Math.random() - 0.5);
    }
    predict(state) {
        const input = this.state_to_vector(state);
        const hidden = [];
        for (let i = 0; i < this.num_of_hidden_neurons; i++) {
            hidden[i] = this.sigmoid(this.dotProduct(input, this.W_hidden_input[i]) + this.bias_hidden_layer[i]);
        }
        const logits = [];
        for (let i = 0; i < this.num_outputs; i++) {
            logits[i] = this.dotProduct(hidden, this.W_hidden_output[i]) + this.bias_output_layer[i];
        }
        const outputs = this.softmax(logits);
        let bestAction = 'none';
        let maxScore = -Infinity;
        let secondScore = -Infinity;
        let bestIndex = -1;
        let secondIndex = -1;
        for (let i = 0; i < outputs.length; i++) {
            if (outputs[i] > maxScore) {
                secondScore = maxScore;
                secondIndex = bestIndex;
                maxScore = outputs[i];
                bestIndex = i;
            }
            else if (outputs[i] > secondScore) {
                secondScore = outputs[i];
                secondIndex = i;
            }
        }
        const delta = 0.2;
        if (secondIndex !== -1 && maxScore - secondScore < delta) {
            const combo = [actions[bestIndex], actions[secondIndex]].sort().join('-');
            switch (combo) {
                case 'up-right':
                    bestAction = 'up-right';
                    break;
                case 'up-left':
                    bestAction = 'up-left';
                    break;
                case 'down-right':
                    bestAction = 'down-right';
                    break;
                case 'down-left':
                    bestAction = 'down-left';
                    break;
                default:
                    bestAction = actions[bestIndex];
                    break;
            }
        }
        else
            bestAction = actions[bestIndex];
        return bestAction;
    }
    ;
    single_fit(state, correctAction) {
        const input = this.state_to_vector(state);
        const hidden = [];
        for (let i = 0; i < this.num_of_hidden_neurons; i++) {
            hidden[i] = this.sigmoid(this.dotProduct(input, this.W_hidden_input[i]) + this.bias_hidden_layer[i]);
        }
        const logits = [];
        for (let i = 0; i < this.num_outputs; i++) {
            logits[i] = this.dotProduct(hidden, this.W_hidden_output[i]) + this.bias_output_layer[i];
        }
        const outputs = this.softmax(logits);
        // stochastic gradient descent: (is stochastic because I update after every pass)
        let target = [];
        for (let i = 0; i < this.num_outputs; i++) {
            target[i] = actions[i] === correctAction ? 1 : 0;
        }
        //back propagation:
        const delta_output = [];
        for (let k = 0; k < this.num_outputs; k++) {
            delta_output[k] = target[k] - outputs[k];
        }
        const delta_hidden = [];
        for (let j = 0; j < this.num_of_hidden_neurons; j++) {
            let sum = 0;
            for (let k = 0; k < this.num_outputs; k++) {
                sum += delta_output[k] * this.W_hidden_output[k][j];
            }
            delta_hidden[j] = hidden[j] * (1 - hidden[j]) * sum; //sigmoid derivative * sum
        }
        // Update weights:
        for (let k = 0; k < this.num_outputs; k++) {
            for (let j = 0; j < this.num_of_hidden_neurons; j++) {
                this.W_hidden_output[k][j] += this.learning_rate * delta_output[k] * hidden[j];
            }
            this.bias_output_layer[k] += this.learning_rate * delta_output[k];
        }
        for (let j = 0; j < this.num_of_hidden_neurons; j++) {
            for (let i = 0; i < this.num_inputs; i++) {
                this.W_hidden_input[j][i] += this.learning_rate * delta_hidden[j] * input[i];
            }
            this.bias_hidden_layer[j] += this.learning_rate * delta_hidden[j];
        }
    }
    fit(states, correctActions, epochs = this.n_iter) {
        for (let epoch = 0; epoch < epochs; epoch++) {
            for (let i = 0; i < states.length; i++) {
                this.single_fit(states[i], correctActions[i]);
            }
        }
    }
    ;
}
//# sourceMappingURL=predictive_model.js.map