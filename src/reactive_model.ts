/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   reactive_model.ts                                  :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: yohan <yohan@student.42.fr>                +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/09/28 09:30:00 by yohan             #+#    #+#             */
/*   Updated: 2025/09/28 09:20:52 by yohan            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

export type state = {
  X_pos: number;
  Y_pos: number;
  Z_pos: number;
  Vx: number;
  Vy: number;
  Vz: number;
  X_paddle: number;
  Y_paddle: number;
  paddle_speed: number;
  paddle_width?: number;
  paddle_height: number;
};

export type action =
| 'up'
| 'down'
| 'left'
| 'right'
| 'up-right'
| 'up-left'
| 'down-right'
| 'down-left'
| 'none';

export const actions: action[] = [
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

export type Weights = {
W_hidden_input: number[][];
W_hidden_output: number[][];
bias_hidden_layer: number[];
bias_output_layer: number[];
};

export class reactive_model {
  public num_inputs: number = 10;
  public num_hidden: number = 12;
  public num_outputs: number = 9;
  public learning_rate: number;

  // Neural network weights
  public W_hidden_input: number[][];
  public W_hidden_output: number[][];
  public bias_hidden_layer: number[];
  public bias_output_layer: number[];

  //______________________________________________________________________________//
  //                              Mathematical helpers                            //
  //______________________________________________________________________________//
  
  public dotProduct(a: number[], b: number[]): number {
      if (a.length !== b.length) {
          throw new Error(`Vector length mismatch: ${a.length} vs ${b.length}`);
      }
      let sum = 0;
      for (let i = 0; i < a.length; i++) {
          sum += a[i] * b[i];
      }
      return sum;
  }

  private sigmoid(x: number): number {
      return 1 / (1 + Math.exp(-x));
  }

  private softmax(logits: number[]): number[] {
      const maxLogit = Math.max(...logits);
      const expScores = logits.map(v => Math.exp(v - maxLogit));
      const sumExp = expScores.reduce((a, b) => a + b, 0);
      return expScores.map(v => v / sumExp);
  }

  private stateToVector(state: state): number[] {
      return [
          state.X_pos,
          state.Y_pos,
          state.Z_pos,
          state.Vx,
          state.Vy,
          state.Vz,
          state.X_paddle,
          state.Y_paddle,
          state.paddle_speed,
          state.paddle_height
      ];
  }

  //______________________________________________________________________________//
  //                                Constructor                                   //
  //______________________________________________________________________________//
  
  constructor(learning_rate: number = 0.1, weights?: Weights) {
      this.learning_rate = learning_rate;
      
      if (weights) {
          // Load provided weights
          this.W_hidden_input = weights.W_hidden_input;
          this.W_hidden_output = weights.W_hidden_output;
          this.bias_hidden_layer = weights.bias_hidden_layer;
          this.bias_output_layer = weights.bias_output_layer;
          
          // Verify dimensions
          if (this.W_hidden_input.length !== this.num_hidden) {
              throw new Error(`W_hidden_input rows mismatch: expected ${this.num_hidden}, got ${this.W_hidden_input.length}`);
          }
          if (this.W_hidden_input[0].length !== this.num_inputs) {
              throw new Error(`W_hidden_input cols mismatch: expected ${this.num_inputs}, got ${this.W_hidden_input[0].length}`);
          }
          if (this.W_hidden_output.length !== this.num_outputs) {
              throw new Error(`W_hidden_output rows mismatch: expected ${this.num_outputs}, got ${this.W_hidden_output.length}`);
          }
          if (this.W_hidden_output[0].length !== this.num_hidden) {
              throw new Error(`W_hidden_output cols mismatch: expected ${this.num_hidden}, got ${this.W_hidden_output[0].length}`);
          }
      } else {
          // Random initialization (same as original)
          this.W_hidden_input = Array(this.num_hidden)
              .fill(0)
              .map(() => Array(this.num_inputs).fill(0).map(() => Math.random() - 0.5));
          
          this.W_hidden_output = Array(this.num_outputs)
              .fill(0)
              .map(() => Array(this.num_hidden).fill(0).map(() => Math.random() - 0.5));
          
          this.bias_hidden_layer = Array(this.num_hidden)
              .fill(0)
              .map(() => Math.random() - 0.5);
          
          this.bias_output_layer = Array(this.num_outputs)
              .fill(0)
              .map(() => Math.random() - 0.5);
      }
  }

  //______________________________________________________________________________//
  //                                Prediction                                   //
  //______________________________________________________________________________//
  
  public predict(state: state): action {
      const input = this.stateToVector(state);
      
      // Forward pass through hidden layer
      const hidden: number[] = [];
      for (let i = 0; i < this.num_hidden; i++) {
          const activation = this.dotProduct(input, this.W_hidden_input[i]) + this.bias_hidden_layer[i];
          hidden[i] = this.sigmoid(activation);
      }
      
      // Forward pass through output layer
      const logits: number[] = [];
      for (let i = 0; i < this.num_outputs; i++) {
          logits[i] = this.dotProduct(hidden, this.W_hidden_output[i]) + this.bias_output_layer[i];
      }
      
      const outputs = this.softmax(logits);
      
      // Simple action selection - highest probability wins
      let maxScore = -Infinity;
      let bestIndex = -1;
      for (let i = 0; i < outputs.length; i++) {
          if (outputs[i] > maxScore) {
              maxScore = outputs[i];
              bestIndex = i;
          }
      }
      
      return actions[bestIndex];
  }

  //______________________________________________________________________________//
  //                                Training                                     //
  //______________________________________________________________________________//
  
  public single_fit(state: state, correctAction: action): void {
      const input = this.stateToVector(state);
      
      // Forward pass
      const hidden: number[] = [];
      for (let i = 0; i < this.num_hidden; i++) {
          const activation = this.dotProduct(input, this.W_hidden_input[i]) + this.bias_hidden_layer[i];
          hidden[i] = this.sigmoid(activation);
      }
      
      const logits: number[] = [];
      for (let i = 0; i < this.num_outputs; i++) {
          logits[i] = this.dotProduct(hidden, this.W_hidden_output[i]) + this.bias_output_layer[i];
      }
      
      const outputs = this.softmax(logits);
      
      // Create target vector
      const target: number[] = [];
      for (let i = 0; i < this.num_outputs; i++) {
          target[i] = actions[i] === correctAction ? 1 : 0;
      }

      // Backward pass
      const delta_output: number[] = [];
      for (let k = 0; k < this.num_outputs; k++) {
          delta_output[k] = target[k] - outputs[k];
      }

      const delta_hidden: number[] = [];
      for (let j = 0; j < this.num_hidden; j++) {
          let sum = 0;
          for (let k = 0; k < this.num_outputs; k++) {
              sum += delta_output[k] * this.W_hidden_output[k][j];
          }
          delta_hidden[j] = hidden[j] * (1 - hidden[j]) * sum; // sigmoid derivative
      }
      
      // Update weights
      for (let k = 0; k < this.num_outputs; k++) {
          for (let j = 0; j < this.num_hidden; j++) {
              this.W_hidden_output[k][j] += this.learning_rate * delta_output[k] * hidden[j];
          }
          this.bias_output_layer[k] += this.learning_rate * delta_output[k];
      }
      
      for (let j = 0; j < this.num_hidden; j++) {
          for (let i = 0; i < this.num_inputs; i++) {
              this.W_hidden_input[j][i] += this.learning_rate * delta_hidden[j] * input[i];
          }
          this.bias_hidden_layer[j] += this.learning_rate * delta_hidden[j];
      }
  }
  
  public fit(states: state[], correctActions: action[], epochs: number = 100): void {
      for (let epoch = 0; epoch < epochs; epoch++) {
          for (let i = 0; i < states.length; i++) {
              this.single_fit(states[i], correctActions[i]);
          }
      }
  }

  //______________________________________________________________________________//
  //                                Utilities                                    //
  //______________________________________________________________________________//
  
  public getWeights(): Weights {
      return {
          W_hidden_input: this.W_hidden_input,
          W_hidden_output: this.W_hidden_output,
          bias_hidden_layer: this.bias_hidden_layer,
          bias_output_layer: this.bias_output_layer
      };
  }

  public setWeights(weights: Weights): void {
      this.W_hidden_input = weights.W_hidden_input;
      this.W_hidden_output = weights.W_hidden_output;
      this.bias_hidden_layer = weights.bias_hidden_layer;
      this.bias_output_layer = weights.bias_output_layer;
  }
}