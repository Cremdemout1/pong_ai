/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   reactive_model.ts                                  :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: yohan <yohan@student.42.fr>                +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/09/07 10:00:48 by yohan             #+#    #+#             */
/*   Updated: 2025/09/07 11:23:51 by yohan            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */


export type state = {
    X_pos: number,
    Y_pos: number,
    Z_pos: number,
    Vx: number,
    Vy: number,
    Vz: number,
    X_paddle: number,
    Y_paddle: number,
    paddle_speed: number,
    paddle_width: number,
    paddle_height: number
};

export type state_intercept = {
    X_pos: number,
    Y_pos: number,
    Z_pos: number,
    Vx: number,
    Vy: number,
    Vz: number,
    X_paddle: number,
    Y_paddle: number,
    paddle_speed: number,
    paddle_width: number,
    paddle_height: number,
    time_to_wall_x: number,
    time_to_wall_y: number,
    x_dist_to_paddle: number,
    y_dist_to_paddle: number
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
  W_hidden_input?: number[][];
  W_hidden_output?: number[][];
  bias_hidden_layer?: number[];
  bias_output_layer?: number[];
};
