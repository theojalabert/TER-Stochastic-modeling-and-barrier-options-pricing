# TER - Stochastic modeling and barrier options pricing

**Abstract**

  Over the last ten years, the financial products available on the markets have become particularly complex: after the first generation of derivatives introduced in the 1970s, based on the famous work of Black and Scholes, the second generation of exotic products appeared in the early 1990s. Among them, and this is the subject of this report, are barrier options, whose main characteristic is the dependence of the option’s terminal flow on the full path of the underlying asset’s price.
  
  In this report, we study the different pricing methods for barrier options. We will use the Heston model with jumps, otherwise known as the Bates model, which is a stochastic volatility model used in activating derivatives. We present the analytical solutions of this model before looking at the implementation of the pricer in Python thanks to the Monte-Carlo method using variance reduction methods such as the antithetical variables method. We then calibrate our model with S&P500 data and analyse the time to price and parameter settings. Finally, we study the pricing of barrier options under the binomial tree model.
  
**Résumé**
  Au cours des dix dernières années, les produits financiers disponibles sur les marchés se sont particulièrement complexifiés : après les produits dérivés de première génération introduits dans les années 1970, et sur lesquels s’appuient les célèbres travaux de Black et Scholes, les produits exotiques de deuxième génération ont fait leur apparition au début des années 1990. Parmi eux, et c’est ce dont fait l’objet ce rapport, les options barrières dont la caractéristique principale réside dans la dépendance du flux terminal de l’option en la trajectoire complète du cours de l’actif sous-jacent.

  Dans ce rapport, nous étudions les différentes méthodes de pricing pour les options à barrières. Nous utiliserons le modèles de Heston avec sauts autrement appelé modèle de Bates, notamment utilisé en dérivés actions, qui est un modèle à volatilité stochastique. Nous présentons les solutions analytiques de ce modèle avant de s’intéresser à l’implémentation du pricer en Python grâce à la méthode de Monte-Carlo en utilisant des méthodes de réduction de variance comme celle des variables antithétiques. Ensuite, nous calibrons notre modèle grâce aux données de l’index S&P500 puis analysons les temps de fixations des prix et des paramètres. Pour finir, nous avons étudié le pricing d’option barrières sous le modèle d’arbres binomiaux.


**Key-words**
  Finance , pricing , options , Monte-Carlo method , stochastic modeling , barrier option , underlying asset , Heston model , Bates model , variance reduction , antithetical variables , calibration , S&P500 , Euler-Maruyama discretisation, binomial tree method
