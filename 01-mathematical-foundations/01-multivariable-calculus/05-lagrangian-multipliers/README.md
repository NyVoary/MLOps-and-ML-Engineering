Les Multiplicateurs de Lagrange (Lagrangian Multipliers) ⚖️ sont un outil puissant pour l'optimisation sous contraintes. Nous allons explorer cette méthode pas à pas !🎯 Le Problème : Optimiser sous contrainteEn optimisation classique (comme la descente de gradient), nous cherchons le minimum ou maximum d'une fonction $f(x, y)$ sans aucune restriction.Mais en Machine Learning et en physique, nous avons souvent des contraintes :💰 Maximiser un profit $f(x, y)$ sous la contrainte de ne pas dépasser un budget $g(x, y) = c$.🛡️ Minimiser une erreur de modèle sous la contrainte de régularisation (ex: limiter la taille des poids).💡 L'Intuition GéométriqueImaginons que les courbes de niveau de notre fonction à optimiser $f(x, y)$ forment des lignes (comme sur une carte d'état-major) et que la contrainte $g(x, y) = c$ forme une courbe spécifique.Au point de la contrainte qui atteint le maximum ou minimum de $f$ :La courbe de contrainte est tangente à une courbe de niveau de $f$.Leurs vecteurs gradients ($\nabla f$ et $\nabla g$) sont donc alignés (parallèles) !Ce qui nous donne l'équation fondamentale de Lagrange :$$\nabla f = \lambda \cdot \nabla g$$(où $\lambda$ est le scalaire appelé multiplicateur de Lagrange)🧩 La Fonction Lagrangienne $\mathcal{L}$Pour résoudre ce système facilement, on regroupe tout dans une seule fonction :$$\mathcal{L}(x, y, \lambda) = f(x, y) - \lambda \cdot (g(x, y) - c)$$Trouver le point optimal revient à chercher l'endroit où le gradient de $\mathcal{L}$ s'annule : $\nabla \mathcal{L} = \mathbf{0}$.🧪 Une première étape ensemblePrenons un problème très simple :Fonction à maximiser : $f(x, y) = x + y$Contrainte : $x^2 + y^2 = 1$ (notre point doit rester sur le cercle unité)Pour commencer, calculons séparément les gradients des deux fonctions :Quel est le gradient de $f(x, y)$, soit $\nabla f$ ?Quel est le gradient de la contrainte $g(x, y) = x^2 + y^2$, soit $\nabla g$ ?

C'est parfaitement exact ! 🎯

Pour f(x,y)=x+y, les dérivées partielles sont  
∂x
∂f
​
 =1 et  
∂y
∂f
​
 =1, donc ∇f=( 
1
1
​
 ).

Pour g(x,y)=x 
2
 +y 
2
 , les dérivées partielles sont  
∂x
∂g
​
 =2x et  
∂y
∂g
​
 =2y, donc ∇g=( 
2x
2y
​
 ).

⚙️ Appliquons la condition de Lagrange
L'équation d'alignement des gradients ∇f=λ⋅∇g nous donne le système à deux équations suivant :

1=λ⋅(2x)
1=λ⋅(2y)
Et nous avons toujours notre contrainte initiale :

x 
2
 +y 
2
 =1
💡 À toi d'exprimer x et y
En observant les deux premières équations :

Que peux-tu en déduire sur la relation entre x et y ?

Si tu remplaces cette relation dans la contrainte x 
2
 +y 
2
 =1, quelles valeurs trouves-tu pour x et y ?

Pour résoudre numériquement un problème d'optimisation sous contrainte d'égalité ($g(x, y) = c$) avec la méthode des Multiplicateurs de Lagrange, nous cherchons le point où le gradient de la fonction Lagrangienne $\mathcal{L}(x, y, \lambda) = f(x, y) - \lambda \cdot (g(x, y) - c)$ s'annule complètement :$$\nabla \mathcal{L}(x, y, \lambda) = \mathbf{0}$$

ALGORITHME: Solve_Lagrange_Equality_Constraint

ENTRÉES:
    - f(x, y) : Fonction objectif à optimiser
    - g(x, y) : Fonction de contrainte
    - c       : Valeur cible de la contrainte (g(x, y) = c)
    - point_initial : Estimation du vecteur [x, y, lambda]

SORTIE:
    - [x_opt, y_opt] : Point optimal respectant la contrainte
    - lambda_opt     : Multiplicateur de Lagrange à l'équilibre
    - f_opt          : Valeur maximale/minimale f(x_opt, y_opt)

DÉBUT
    1. Définir le système d'équations non-linéaires F(x, y, lambda) = 0:
       │
       ├── Équation 1 : ∂f/∂x - lambda * (∂g/∂x) = 0   (Gradient f = lambda * Gradient g sur x)
       ├── Équation 2 : ∂f/∂y - lambda * (∂g/∂y) = 0   (Gradient f = lambda * Gradient g sur y)
       └── Équation 3 : g(x, y) - c = 0               (Respect de la contrainte)

    2. Résoudre le système F(x, y, lambda) = [0, 0, 0] 
       (En utilisant un solveur numérique de racines comme Newton-Raphson)

    3. Extraire les solutions [x_opt, y_opt, lambda_opt]

    4. Calculer la valeur optimale : f_opt = f(x_opt, y_opt)

    RETOURNER x_opt, y_opt, lambda_opt, f_opt
FIN

Remarque comment les 3 équations du pseudo-code correspondent exactement à ce que nous avons fait à la main juste avant !Pour notre exemple ($f(x, y) = x + y$ et $g(x, y) = x^2 + y^2 = 1$) :L'équation 1 devient : $1 - \lambda(2x) = 0$L'équation 2 devient : $1 - \lambda(2y) = 0$L'équation 3 devient : $x^2 + y^2 - 1 = 0$