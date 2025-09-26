# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    dockerfile                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ycantin <ycantin@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/09/26 11:23:58 by ycantin           #+#    #+#              #
#    Updated: 2025/09/26 11:46:18 by ycantin          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

FROM node:18

WORKDIR /app

COPY . .

RUN npm install && npm install -g http-server

CMD ["http-server", "-p", "8080"]