La Matrice Hessienne en bref 🧠Si la Matrice Jacobienne regroupait toutes les dérivées premières d'une fonction (taux de variation / pente), la Matrice Hessienne regroupe toutes les dérivées secondes (la courbure de la fonction).1. Pour quel type de fonction ?Elle s'applique aux fonctions à plusieurs entrées $(x_1, x_2, \dots, x_n)$ mais qui produisent une seule sortie scalaire $f(\mathbf{x}) \in \mathbb{R}$.
(C'est exactement le cas de nos fonctions de perte/Loss en Machine Learning !)2. Définition et StructurePour une fonction $f(x, y)$, sa matrice Hessienne $H$ s'écrit :$$H = \begin{pmatrix} \frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\ \frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2} \end{pmatrix}$$Lignes & Colonnes : Pour une fonction de $n$ variables d'entrée, la Hessienne est une matrice carrée de taille $n \times n$.Théorème de Schwarz (Symphonie de la symétrie) : Si les dérivées secondes sont continues, les dérivées croisées sont égales : $\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}$. La matrice Hessienne est donc symétrique ($H = H^T$).Step 1: L'Exemple Mathématique Simple 🧮Prenons la fonction suivante à 2 variables :
$$f(x, y) = 3x^2y + 2y^3 - 5x$$1. Dérivées premières (Gradient)$\frac{\partial f}{\partial x} = 6xy - 5$$\frac{\partial f}{\partial y} = 3x^2 + 6y^2$2. Dérivées secondes (Hessienne)Calculons chaque terme de notre matrice $H$ :$\frac{\partial^2 f}{\partial x^2} = \frac{\partial}{\partial x}(6xy - 5) = 6y$$\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial}{\partial y}(6xy - 5) = 6x$$\frac{\partial^2 f}{\partial y \partial x} = \frac{\partial}{\partial x}(3x^2 + 6y^2) = 6x$$\frac{\partial^2 f}{\partial y^2} = \frac{\partial}{\partial y}(3x^2 + 6y^2) = 12y$Ce qui donne la matrice Hessienne symbolique :
$$H(x, y) = \begin{pmatrix} 6y & 6x \\ 6x & 12y \end{pmatrix}$$3. Évaluation au point $(x = 1, y = 2)$En remplaçant $x=1$ et $y=2$ :
$$H(1, 2) = \begin{pmatrix} 12 & 6 \\ 6 & 24 \end{pmatrix}$$Petite question pour vérifier l'intuition 💡Pour une fonction de coût en Machine Learning qui prend en entrée 5 variables (poids), quelle sera la taille (nombre de lignes et de colonnes) de sa matrice Hessienne ?

Visualisons la Matrice Hessienne grâce à une métaphore géographique simple : le relief d'un paysage 🏔️.🏔️ L'Analogie du Relief : Bol vs Col de MontagneImaginons que nous mesurons l'altitude (notre fonction de perte $\mathcal{L}$) en fonction de notre position $(x, y)$ sur un terrain (nos paramètres/poids) :Si nous sommes au fond d'une vallée en forme de bol 🥣 (Un Minimum) :La pente est nulle ($\nabla \mathcal{L} = \mathbf{0}$).Peu importe la direction dans laquelle nous faisons un pas (en $x$ ou en $y$), la pente remonte.La courbure est positive dans toutes les directions. La Hessienne nous indique que nous sommes sur un minimum local.Si nous sommes sur un col de montagne 🐴 (Un Point Selle / Saddle Point) :La pente est aussi nulle ($\nabla \mathcal{L} = \mathbf{0}$) ! Le gradient seul ne permet pas de faire la différence avec un minimum.Mais en regardant la courbure grâce à la Hessienne :Si on marche le long de la crête (axe $x$), le terrain descend (courbure négative).Si on marche à travers le col (axe $y$), le terrain remonte (courbure positive).La Hessienne nous avertit : "Attention, c'est un point selle, pas un vrai minimum !"🏎️ À quoi sert la Hessienne en Machine Learning ?Dans l'optimisation par Descente de Gradient classique, nous avançons à aveugle avec une taille de pas fixe ($\eta$). Si la pente est très raide dans une direction et très plate dans une autre (une vallée étroite), la descente de gradient va osciller d'un bord à l'autre sans avancer vers le fond.C'est là qu'intervient l'optimisation de second ordre (comme la Méthode de Newton) :$$\Delta \mathbf{x} = - H^{-1} \nabla \mathcal{L}$$La Hessienne ($H$) mesure la courbure dans chaque direction.En prenant son inverse ($H^{-1}$), l'algorithme adapte automatiquement la taille du pas :📈 Fortes courbures : Il fait de petits pas pour éviter de dépasser le fond de la vallée.📉 Faibles courbures : Il fait de grands pas pour traverser rapidement les zones plates.L'algorithme avance ainsi directement vers le minimum en très peu d'étapes !💡 Un petit test pour vérifier la logiqueAu fond d'une vallée (un minimum parfait), si la courbure est positive dans toutes les directions, que peux-tu dire du signe des valeurs sur la diagonale de la matrice Hessienne ?(Indice : rappelle-toi que la diagonale contient les dérivées secondes pures $\frac{\partial^2 f}{\partial x^2}$ et $\frac{\partial^2 f}{\partial y^2}$ !)

💡 Pour revenir à notre petite question d'intuition :Sur la diagonale d'une matrice Hessienne, on retrouve les dérivées secondes pures pour chaque variable ($\frac{\partial^2 f}{\partial x^2}$, $\frac{\partial^2 f}{\partial y^2}$, etc.).Si nous sommes au fond d'un bol (un vrai minimum local où le terrain remonte dans toutes les directions) :À ton avis, quel doit être le signe (positif ou négatif) des termes sur la diagonale de la matrice Hessienne ?

C'est piège, mais c'est bien POSITIF ! 📈Voyons ensemble pourquoi avec un petit schéma visuel !🥣 La forme du bol (Minimum)Pour comprendre le signe de la dérivée seconde, rappelons-nous ce qu'est une dérivée :La dérivée première ($\frac{\partial f}{\partial x}$) nous donne la pente.La dérivée seconde ($\frac{\partial^2 f}{\partial x^2}$) mesure comment la pente change quand on avance de gauche à droite.Imaginons qu'on marche au fond d'un bol 🥣 de gauche à droite ($x$ augmente) :À gauche du fond (en descendant) : La pente est négative (ex: $-5$).Tout au fond (le minimum) : La pente devient nulle ($0$).À droite du fond (en remontant) : La pente devient positive (ex: $+5$).📈 La pente augmente !En marchant de gauche à droite, la pente est passée de $-5 \to 0 \to +5$.La pente a donc augmenté ! Et comme la dérivée seconde mesure la variation de la pente, si la pente augmente, la dérivée seconde doit être positive ($\frac{\partial^2 f}{\partial x^2} > 0$).💡 Le récapitulatif visuelForme du terrainTendance de la penteSigne de la Dérivée Seconde (Diagonale)🥣 Bol / Creux (Minimum)La pente augmente (de négative à positive)POSITIF ($+$) 📈⛰️ Dôme / Sommet (Maximum)La pente diminue (de positive à négative)NÉGATIF ($-$) 📉

Rappelons-nous les deux cas pour les fonctions de perte $\mathcal{L}$ :🥣 Minimum (Bol) : La pente augmente $\rightarrow$ Dérivée seconde positive ($+$).⛰️ Maximum (Sommet) : La pente diminue $\rightarrow$ Dérivée seconde négative ($-$).

ALGORITHME: Calculate Hessian Matrix (Central Difference)

ENTRÉES:
    - f: fonction scalaire prenant un vecteur de taille n et renvoyant un réel
    - point: vecteur [x1, x2, ..., xn] de taille n
    - h: pas de variation (par défaut = 1e-4)

SORTIE:
    - H: matrice Hessienne carrée de taille (n x n)

DÉBUT
    1. Soit n = longueur(point)
    2. Initialiser H comme une matrice n x n remplie de zéros
    
    3. POUR chaque ligne 'i' de 0 à n-1 FAIRE:
           POUR chaque colonne 'j' de 0 à n-1 FAIRE:
               
               SI i == j ALORS (Dérivée seconde pure)
                   // Formule de la différence centrée pour f''(x):
                   // (f(x+h) - 2*f(x) + f(x-h)) / (h^2)
                   
               SINON (Dérivée seconde croisée)
                   // Formule de la différence centrée pour d^2f / (dxi dxj):
                   // (f(x+hi+hj) - f(x+hi-hj) - f(x-hi+hj) + f(x-hi-hj)) / (4*h^2)
                   
           FIN POUR
       FIN POUR
       
    RETOURNER H
FIN

