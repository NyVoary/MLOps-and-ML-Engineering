# Partial Derivatives and Gradient Vector

## 1. The Partial Derivative 🔍

When a function depends on several variables — for example $f(x, y)$ — the **partial derivative** measures how the function changes when we vary **a single variable**, keeping all others constant.

### Notation

The partial derivative with respect to $x$ is denoted:

$$\frac{\partial f}{\partial x} \quad \text{or} \quad f'_x$$

The partial derivative with respect to $y$ is denoted:

$$\frac{\partial f}{\partial y} \quad \text{or} \quad f'_y$$

### Worked example

Consider the following simplified cost function:

$$f(x, y) = 3x^2 + 2xy + y^3$$

To compute $\dfrac{\partial f}{\partial x}$, we treat $y$ as a plain constant (like any fixed number: 2, 5, etc.):

$$\frac{\partial f}{\partial x} = 6x + 2y$$

To compute $\dfrac{\partial f}{\partial y}$, we treat $x$ as a constant:

$$\frac{\partial f}{\partial y} = 2x + 3y^2$$

## 2. The Gradient Vector 🧭

The **gradient** gathers all the partial derivatives of a function into a single vector.

### Definition

For a function $f(x, y)$, the gradient vector (denoted $\nabla f$, read "nabla f") is:

$$\nabla f(x, y) = \begin{pmatrix} \dfrac{\partial f}{\partial x} \\[6pt] \dfrac{\partial f}{\partial y} \end{pmatrix}$$

### Key geometric property

The gradient vector has two characteristics that are crucial for Machine Learning:

- **Direction**: it always points toward the direction where the function increases the fastest.
- **Norm (magnitude)**: its size indicates how steep the slope is.

> 💡 **The link to optimization (Gradient Descent):**
> To reduce an error (minimize a cost function), we move in the direction **opposite** to the gradient: $-\nabla f$.

### Your turn! ✏️

Let's apply this theory directly to a small practical case before moving to Python.

Consider the following function:

$$f(x, y) = x^2 y + 4y^2$$

- What is the partial derivative $\dfrac{\partial f}{\partial x}$?
- What is the partial derivative $\dfrac{\partial f}{\partial y}$?

Try to work out both calculations!

## 3. Script structure

To keep software-engineering rigor (with doctests, typing, and docstrings):

1. **Analytical (exact) approach**: a Python function that computes the exact value of the function and its gradient vector at point $(x, y)$.
2. **Numerical approach (finite differences)**: a generic function that computes the gradient of any function using the centered finite-difference approximation:

$$\frac{\partial f}{\partial x} \approx \frac{f(x + h, y) - f(x - h, y)}{2h}$$

3. **Validation**: comparing the exact (analytical) value against the numerical approximation to verify our implementation is correct.

### The intuition behind this pseudo-code 💡

- **The small variation $h$**: we take a very small step (e.g. $10^{-5}$) to zoom in on the curve around the point $(x, y)$.
- **Isolating the variable**:
  - To get $\dfrac{\partial f}{\partial x}$, we evaluate the function slightly to the right $f(x+h, y)$ and slightly to the left $f(x-h, y)$. We don't touch $y$!
  - To get $\dfrac{\partial f}{\partial y}$, it's the opposite: $x$ stays fixed, and we vary $y$.
- **Computing the slope**: the difference $(f_{plus} - f_{minus})$ divided by the total distance $2h$ gives us the exact slope of the tangent (the partial derivative).

### Two ways to compute this in software

| Type of computation | What the program does | Example input | Example output | Main Python tool |
|---|---|---|---|---|
| **Numerical** | Computes exact or approximated values at a specific point $(x, y)$. | $x=1, y=2$ | The number $17.0$ | NumPy |
| **Symbolic** | Manipulates variables like on paper to find the mathematical expression of the derivative. | $f(x, y) = x^2 y + 4y^2$ | The expression $x^2 + 8y$ | SymPy |

### Comparing the methods

| Method | How it works | Advantage | Drawback | Used where? |
|---|---|---|---|---|
| **Symbolic** (SymPy) | Manipulates algebraic expressions to produce a formal formula. | Gives the exact equation $2xy$. | Very slow, requires a substitution step to evaluate at a point. | Theoretical math, formal computation, proof verification. |
| **Numerical** (NumPy) | Uses $\dfrac{f(x+h) - f(x-h)}{2h}$. | Very simple to implement. | Approximation (round-off errors), requires evaluating $f$ multiple times per parameter. | Gradient validation tests ("Gradient Checking"). |
| **Delegated / Automatic** (PyTorch / Autograd) | Computes the exact gradient value at a given point by building a computation graph. | Ultra-fast, exact up to machine precision, ideal for millions of parameters. | Requires a dedicated autodiff framework. | All of modern Deep Learning (Backpropagation). |

## 4. A note on the step size $h = 10^{-5}$

In numerical finite-difference computation, the choice of $h = 10^{-5}$ (i.e. `1e-5`) is an ideal compromise for balancing two opposing types of computer error.

### The two errors at war ⚔️

**A. Truncation Error**

- Theory: mathematically, the smaller $h$ is, the closer the formula $\dfrac{f(x+h) - f(x-h)}{2h}$ gets to the true derivative.
- Problem: if $h$ is too large (e.g. $h = 0.1$), the tangent approximation is very inaccurate because the curve varies too much over that interval.

**B. Round-off Error (Floating-Point Limitation)**

- Problem: computers store decimal numbers with limited precision (typically 64 bits in Python, i.e. about 15 to 17 significant digits).
- Consequence: if $h$ is too small (e.g. $h = 10^{-15}$), the difference $f(x+h) - f(x-h)$ becomes a number extremely close to zero. The computer then loses precision when subtracting two nearly identical numbers (a phenomenon called **catastrophic cancellation**), and then dividing by a tiny $2h$ makes it worse.

### Why exactly $10^{-5}$ or $10^{-8}$? ⚖️

For a standard floating-point representation (64-bit double precision, with $\epsilon \approx 10^{-16}$), for the centered finite-difference formula, the theoretical optimal error is reached when:

$$h \approx \sqrt[3]{\epsilon} \approx \sqrt[3]{10^{-16}} \approx 10^{-5.3}$$

This is why $h = 10^{-5}$ (or $10^{-6}$) is the standard default setting in engineering: it minimizes both the geometric error of the curve and the round-off error of the computer. 💡

---

## Pseudo-code

### Pseudo-code: Partial Derivative (Central Difference)

```
ALGORITHM: Calculate Partial Derivative

INPUT:
    - f: scalar function f(x, y)
    - point: vector containing [x, y]
    - var_index: index of the variable to differentiate (0 for x, 1 for y)
    - h: step size (default = 1e-5)

OUTPUT:
    - df_dvar: partial derivative value at the given point

BEGIN
    1. Create a copy of 'point' named 'point_plus'
    2. Create a copy of 'point' named 'point_minus'

    3. Add 'h' to 'point_plus' at position 'var_index'      // (e.g., [x + h, y] if var_index == 0)
    4. Subtract 'h' from 'point_minus' at position 'var_index' // (e.g., [x - h, y] if var_index == 0)

    5. Evaluate f_plus = f(point_plus)
    6. Evaluate f_minus = f(point_minus)

    7. Compute df_dvar = (f_plus - f_minus) / (2 * h)

    RETURN df_dvar
END
```

### Pseudo-code: Gradient Vector

```
ALGORITHM: Calculate Gradient Vector

INPUT:
    - f: scalar function
    - point: vector [x1, x2, ..., xn]
    - h: step size

OUTPUT:
    - grad: vector containing all partial derivatives [df/dx1, df/dx2, ..., df/dxn]

BEGIN
    1. Initialize 'grad' array of the same size as 'point' with zeros
    2. FOR each index 'i' from 0 to length(point) - 1 DO:
           grad[i] = Calculate Partial Derivative(f, point, var_index=i, h)
       END FOR

    RETURN grad
END
```

### Pseudo-code: Interactive Partial Derivatives & Gradient Calculator

```
ALGORITHM: Interactive Partial Derivatives & Gradient Calculator

INPUT:
    - expr_str: mathematical expression entered by user as text (e.g., "x**2 * y + 4*y**2")
    - x_val: numeric float value for x
    - y_val: numeric float value for y
    - h: step size for numerical derivative (default = 1e-5)

OUTPUT:
    - Symbolic partial derivatives (text formulas via SymPy)
    - Numeric values of partial derivatives evaluated at (x_val, y_val) via NumPy
    - Gradient Vector: ∇f(x, y) = [df/dx, df/dy]

BEGIN
    1. Define symbolic variables 'x' and 'y' using SymPy
    2. Parse 'expr_str' into a symbolic expression f_expr

    3. Compute SYMBOLIC partial derivatives (for display purpose):
          df_dx_expr = diff(f_expr, x)
          df_dy_expr = diff(f_expr, y)

    4. Convert symbolic expression f_expr into a fast NumPy-compatible function:
          f_num = lambdify((x, y), f_expr, 'numpy')

    5. Compute NUMERICAL partial derivatives at point (x_val, y_val) using central differences with NumPy:
          df_dx_num = (f_num(x_val + h, y_val) - f_num(x_val - h, y_val)) / (2 * h)
          df_dy_num = (f_num(x_val, y_val + h) - f_num(x_val, y_val - h)) / (2 * h)

    6. Construct the Gradient Vector with NumPy:
          gradient_vector = array([df_dx_num, df_dy_num])

    7. Display symbolic formulas (SymPy), numerical evaluations (NumPy), and the Gradient Vector
END
```



La Matrice Jacobienne en bref 🧠Jusqu'à présent, nous avons travaillé avec des fonctions qui prennent plusieurs entrées $(x, y)$ et donnent une seule sortie (un nombre scalaire) : $f(x,y) \in \mathbb{R}$. Le gradient regroupe alors toutes les dérivées partielles dans un seul vecteur.La Matrice Jacobienne est tout simplement la généralisation du gradient pour les fonctions à plusieurs entrées ET plusieurs sorties :$$\mathbf{F}(x, y) = \begin{pmatrix} f_1(x, y) \\ f_2(x, y) \end{pmatrix}$$Chaque ligne de la matrice Jacobienne représente le gradient d'une des fonctions de sortie :$$\mathbf{J} = \begin{pmatrix} \frac{\partial f_1}{\partial x} & \frac{\partial f_1}{\partial y} \\ \frac{\partial f_2}{\partial x} & \frac{\partial f_2}{\partial y} \end{pmatrix}$$

Step 1: L'Exemple Mathématique Simple 📐Imaginons une fonction $\mathbf{F}$ qui prend un point à deux dimensions $(x, y)$ en entrée et renvoie un vecteur à deux dimensions en sortie :$$\mathbf{F}(x, y) = \begin{pmatrix} f_1(x, y) \\ f_2(x, y) \end{pmatrix} = \begin{pmatrix} x^2 \cdot y \\ x + 3y \end{pmatrix}$$1. Formule Générale de la Matrice JacobienneLa matrice Jacobienne $J$ regroupe toutes les dérivées partielles. Chaque ligne correspond à une fonction de sortie ($f_1, f_2$), et chaque colonne à une variable d'entrée ($x, y$) :$$J = \begin{pmatrix} \frac{\partial f_1}{\partial x} & \frac{\partial f_1}{\partial y} \\ \frac{\partial f_2}{\partial x} & \frac{\partial f_2}{\partial y} \end{pmatrix}$$2. Calcul des Dérivées Partielles SymboliquesCalculons chaque élément un par un :$\frac{\partial f_1}{\partial x} = 2xy$$\frac{\partial f_1}{\partial y} = x^2$$\frac{\partial f_2}{\partial x} = 1$$\frac{\partial f_2}{\partial y} = 3$Ce qui nous donne la matrice symbolique :$$J(x, y) = \begin{pmatrix} 2xy & x^2 \\ 1 & 3 \end{pmatrix}$$3. Évaluation en un Point PrécisÉvaluons maintenant cette Jacobienne au point $(x = 2, y = 3)$ :$2xy = 2 \cdot 2 \cdot 3 = 12$$x^2 = 2^2 = 4$La matrice Jacobienne numérique au point $(2, 3)$ est donc :$$J(2, 3) = \begin{pmatrix} 12 & 4 \\ 1 & 3 \end{pmatrix}$$Petite question pour vérifier l'intuition 💡D'après cette matrice $J(2, 3) = \begin{pmatrix} 12 & 4 \\ 1 & 3 \end{pmatrix}$ :Si l'on augmente très légèrement la valeur de $x$ de $+0.01$ (tout en gardant $y=3$ fixe), de combien environ la première sortie $f_1$ va-t-elle augmenter ?(Indice : regarde l'élément à la 1ère ligne, 1ère colonne !)

Voici un rappel rapide de ce que nous avions :Le point de départ : Nous avons évalué notre matrice au point $(x = 2, y = 3)$.Le résultat obtenu :$$J(2, 3) = \begin{pmatrix} 12 & 4 \\ 1 & 3 \end{pmatrix}$$Ce que signifie chaque case :Première ligne : Concerne la première sortie $f_1$.Première colonne : Concerne la variable d'entrée $x$.Le chiffre $12$ (en haut à gauche) est donc la dérivée partielle $\frac{\partial f_1}{\partial x} = 12$.Qu'est-ce que ce $12$ veut dire en pratique ? 📈La dérivée partielle indique le taux de variation (la sensibilité). Un chiffre de $12$ signifie que :Si tu avances d'un tout petit pas dans la direction de $x$, la valeur de $f_1$ augmente 12 fois plus vite que ce petit pas.Si on augmente $x$ de $+0.01$ (en gardant $y$ fixe) :L'augmentation de $f_1$ sera d'environ : $12 \times 0.01 = +0.12$.

En effet, dans un cours de géométrie ou d'espace de Banach, on la présente souvent sous son angle le plus pur : une application linéaire qui donne la meilleure approximation locale d'une fonction différentiable entre deux espaces vectoriels.C'est exactement la même chose, mais le contexte d'utilisation change :En Géométrie / Analyse globale 📐 : La Jacobienne te sert à mesurer comment une transformation déforme l'espace (changement de variables, calcul de volumes via le déterminant jacobien, déformation de surfaces).En Machine Learning 🤖 : Elle sert à mesurer comment chaque sortie du modèle réagit à chaque entrée.Prenons un exemple rapide en ML pour faire le lien :Si une couche de réseau de neurones prend un vecteur d'entrée $X$ à 3 dimensions et ressort un vecteur $Y$ à 2 dimensions, la matrice Jacobienne contient tous les taux de variation nécessaires pour propagée l'erreur à l'envers (la backpropagation).

Pour une fonction avec 3 variables d'entrée et 2 sorties, la matrice Jacobienne aura bien :2 lignes (une pour chaque fonction de sortie $f_1, f_2$)3 colonnes (une pour chaque variable d'entrée $x_1, x_2, x_3$)Ce qui nous donne une matrice de taille $2 \times 3$ :$$J = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \frac{\partial f_1}{\partial x_3} \\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \frac{\partial f_2}{\partial x_3} \end{pmatrix}$$La règle d'or à retenir 📐Pour toute fonction $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$ (avec $n$ entrées et $m$ sorties) :La Matrice Jacobienne est de taille $m \times n$ ($m$ lignes, $n$ colonnes).

Voici l'illustration concrète de la Matrice Jacobienne dans un réseau de neurones 🤖.Le Scénario : Une couche de réseau de neuronesImaginons une couche intermédiaire d'un réseau de neurones dense (Fully Connected Layer).Les Entrées ($x_1, x_2, x_3$) : Le vecteur des valeurs transmises par la couche précédente (par exemple, 3 caractéristiques extraites d'une image).Les Sorties ($z_1, z_2$) : Le vecteur produit par cette couche (2 neurones de sortie).Chaque neurone calcule une somme pondérée par ses poids $w$ et un biais $b$ :$$z_1 = w_{11}x_1 + w_{12}x_2 + w_{13}x_3 + b_1$$$$z_2 = w_{21}x_1 + w_{22}x_2 + w_{23}x_3 + b_2$$À quoi sert la Matrice Jacobienne ici ? 🎯Pendant la phase d'entraînement, le réseau commet une erreur au niveau de ses sorties (mesurée par une fonction de perte $\mathcal{L}$). Pour ajuster les poids des couches précédentes, l'algorithme de Backpropagation doit savoir :"Si je modifie légèrement l'entrée $x_1$, quel impact cela a-t-il sur la sortie $z_1$ ET sur la sortie $z_2$ ?"Cette information précise est stockée dans la Matrice Jacobienne de la couche ($\mathbf{J} = \frac{\partial \mathbf{z}}{\partial \mathbf{x}}$) :$$J = \begin{pmatrix} \frac{\partial z_1}{\partial x_1} & \frac{\partial z_1}{\partial x_2} & \frac{\partial z_1}{\partial x_3} \\ \frac{\partial z_2}{\partial x_1} & \frac{\partial z_2}{\partial x_2} & \frac{\partial z_2}{\partial x_3} \end{pmatrix} = \begin{pmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \end{pmatrix}$$Constat clé : Pour une couche linéaire $Z = W X + B$, la matrice Jacobienne par rapport à l'entrée $X$ n'est autre que la matrice des poids $W$ elle-même !C'est pour cela que les bibliothèques comme PyTorch ou TensorFlow effectuent continuellement des multiplications de matrices Jacobiennes sous le capot (via la règle de dérivation en chaîne / Chain Rule) pour transmettre l'erreur de couche en couche.

ALGORITHM: Calculate Jacobian Matrix (Central Difference)

INPUT:
    - F: vector-valued function returning an array of size m
    - point: vector [x1, x2, ..., xn] of size n
    - h: step size (default = 1e-5)

OUTPUT:
    - J: Jacobian matrix of size (m x n)

BEGIN
    1. Evaluate F_ref = F(point)
    2. Set m = length(F_ref)
    3. Set n = length(point)
    
    4. Initialize J as a zero matrix of shape (m, n)
    
    5. FOR each column index 'j' from 0 to n - 1 DO:
           Create copy 'point_plus' = point
           Create copy 'point_minus' = point
           
           point_plus[j] = point_plus[j] + h
           point_minus[j] = point_minus[j] - h
           
           F_plus = F(point_plus)
           F_minus = F(point_minus)
           
           // Compute numerical partial derivative for all outputs with respect to variable j
           J[:, j] = (F_plus - F_minus) / (2 * h)
       END FOR
       
    RETURN J
END

L'algorithme de Backpropagation (ou rétropropagation de l'erreur) est le moteur principal qui permet aux réseaux de neurones d'apprendre ! 🧠Pour bien le comprendre, c'est tout simplement l'application pratique de la dérivation en chaîne (Chain Rule) du calcul multivariable pour ajuster les poids d'un réseau.Voyons ensemble comment il fonctionne pas à pas. Nous utiliserons des questions guidées pour construire une intuition solide ! 🚀🔄 Les 3 étapes du cycle d'apprentissageL'apprentissage d'un réseau de neurones se fait en boucle à travers trois grandes phases :La Passe Avant (Forward Pass) ➡️ : Les données d'entrée traversent le réseau de couche en couche pour produire une prédiction.Le Calcul de la Perte (Loss Calculation) 🎯 : Une fonction de coût (ex: l'erreur quadratique) compare la prédiction du réseau avec la vraie valeur cible pour mesurer l'erreur globale.La Passe Arrière (Backpropagation) ⬅️ : C'est ici qu'intervient l'algorithme ! L'erreur calculée à la sortie est propagée en sens inverse à travers tout le réseau.⚙️ Le rôle de la BackpropagationPendant cette passe arrière, la Backpropagation calcule le gradient de la fonction de perte par rapport à chaque poids $w$ et chaque biais $b$ du réseau.En termes simples, elle répond à la question :"Si je modifie légèrement ce poids particulier $w$ au début ou au milieu du réseau, de combien l'erreur globale va-t-elle augmenter ou diminuer ?"Grâce à ces gradients, un algorithme d'optimisation (comme la Descente de Gradient) sait exactement dans quelle direction et de combien ajuster chaque poids pour réduire l'erreur lors des prochaines prédictions ! 📉💡 Un exemple visuel simpleImaginons une chaîne très simple avec 3 étapes :$$\text{Entrée } x \longrightarrow \text{Neurone (Poids } w) \longrightarrow \text{Prédiction } \hat{y} \longrightarrow \text{Erreur } \mathcal{L}$$Pour savoir comment le poids $w$ influence l'erreur finale $\mathcal{L}$, la règle de dérivation en chaîne nous dit de multiplier les sensibilités locales :$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \times \frac{\partial \hat{y}}{\partial w}$$$\frac{\partial \mathcal{L}}{\partial \hat{y}}$ : Comment l'erreur change selon la prédiction.$\frac{\partial \hat{y}}{\partial w}$ : Comment la prédiction change selon le poids.Dans un réseau profond avec plusieurs couches et plusieurs neurones, ces calculs de sensibilité locale prennent la forme de multiplications de Matrices Jacobiennes !🧪 Une petite question pour vérifier ton intuitionImaginons que nous venons de calculer la passe arrière pour un poids $w$. La dérivée nous donne :$$\frac{\partial \mathcal{L}}{\partial w} = +5$$Cela signifie qu'augmenter le poids $w$ fait augmenter l'erreur globale.Pour diminuer l'erreur lors de l'étape d'ajustement, devrions-nous augmenter ou diminuer la valeur de ce poids $w$ ?

Oui, exactement ! 📉 On doit la diminuer.💡 Pourquoi ?Si la dérivée est positive ($\frac{\partial \mathcal{L}}{\partial w} = +5$), cela signifie que le poids $w$ et l'erreur $\mathcal{L}$ évoluent dans le même sens :↗️ Si on augmente $w$, l'erreur augmente.↘️ Si on diminue $w$, l'erreur diminue.Comme notre but en Machine Learning est de minimiser l'erreur 🎯, nous devons aller dans le sens inverse du gradient !📐 La règle de mise à jour (Descente de Gradient)C'est pour cela que dans la formule de mise à jour de la Descente de Gradient, on met un signe MOINS :$$w_{\text{nouveau}} = w_{\text{actuel}} - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}$$(où $\eta$ est le taux d'apprentissage / learning rate, un tout petit nombre positif comme $0.01$)🚀 En résuméPasse Avant (Forward Pass) : On calcule la prédiction et l'erreur ➡️Backpropagation : On calcule les gradients (Matrices Jacobiennes) pour connaître l'impact de chaque poids sur l'erreur ⬅️Mise à jour : On ajuste les poids dans le sens opposé du gradient pour réduire l'erreur 📉

L'algorithme de la Descente de Gradient 📉 est une méthode d'optimisation fondamentale utilisée pour trouver le minimum d'une fonction de coût (ou fonction de perte $\mathcal{L}$).En termes simples, c'est comme essayer de descendre une montagne dans le brouillard 🌫️ : à chaque pas, tu regardes la pente sous tes pieds et tu fais un pas dans la direction la plus descendante.Voici un rappel synthétique de ses composants clés et de sa formule :🧱 Les 3 éléments de baseLa Fonction de Coût ($\mathcal{L}$) : Elle mesure l'erreur globale de notre modèle. Notre objectif est de rendre cette erreur la plus petite possible 🎯.Le Gradient ($\nabla \mathcal{L}$) : C'est le vecteur qui contient toutes les dérivées partielles de la fonction de coût. Il pointe toujours dans la direction où la fonction augmente le plus vite.Le Taux d'Apprentissage ($\eta$ / Learning Rate) : Un petit nombre positif (ex: $0.01$) qui contrôle la taille du pas que l'on fait à chaque itération.📐 La Formule de Mise à JourPour ajuster un poids $w$ (ou l'ensemble des paramètres), on applique cette règle à chaque étape :$$w_{\text{nouveau}} = w_{\text{actuel}} - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}$$Le signe MOINS est crucial : comme le gradient montre où la fonction monte, on prend sa direction opposée pour la faire descendre.🔄 Les Étapes de l'AlgorithmeInitialiser les poids avec des valeurs de départ (souvent aléatoires).Calculer la prédiction et l'erreur actuelle (Passe Avant / Forward Pass).Calculer le gradient de l'erreur par rapport à chaque poids (Backpropagation).Mettre à jour chaque poids en utilisant la formule ci-dessus.Répéter les étapes 2 à 4 jusqu'à ce que l'erreur ne diminue plus (convergence).

Reprenons l'exemple 2 pas à pas pour que tout devienne limpide.1. La couche de neurones (Forward Pass) ➡️Imaginons une simple couche de réseau de neurones :Entrées ($X$) : Un vecteur de 3 valeurs, par exemple $X = [x_1, x_2, x_3]^T$.Sorties ($Z$) : Un vecteur de 2 neurones, $Z = [z_1, z_2]^T$.Pour calculer ces sorties, chaque neurone multiplie les entrées par ses propres poids ($W$) et ajoute un biais ($b$) :$$z_1 = \mathbf{w_{11}}x_1 + \mathbf{w_{12}}x_2 + \mathbf{w_{13}}x_3 + b_1$$$$z_2 = \mathbf{w_{21}}x_1 + \mathbf{w_{22}}x_2 + \mathbf{w_{23}}x_3 + b_2$$2. Le calcul de la Jacobienne 📐Rappelons-nous : la matrice Jacobienne regroupe toutes les dérivées partielles de chaque sortie par rapport à chaque entrée.Sa taille est de 2 lignes (les sorties $z_1, z_2$) et 3 colonnes (les entrées $x_1, x_2, x_3$) :$$J = \begin{pmatrix} \frac{\partial z_1}{\partial x_1} & \frac{\partial z_1}{\partial x_2} & \frac{\partial z_1}{\partial x_3} \\ \frac{\partial z_2}{\partial x_1} & \frac{\partial z_2}{\partial x_2} & \frac{\partial z_2}{\partial x_3} \end{pmatrix}$$Calculons chaque élément de la 1ère ligne (dérivées de $z_1$) :$\frac{\partial z_1}{\partial x_1} = w_{11}$ (car la dérivée de $w_{11}x_1$ par rapport à $x_1$ est simplement $w_{11}$, et tout le reste compte comme constante)$\frac{\partial z_1}{\partial x_2} = w_{12}$$\frac{\partial z_1}{\partial x_3} = w_{13}$Si on fait la même chose pour la 2ème ligne ($z_2$), on obtient :$$J = \begin{pmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \end{pmatrix} = W$$3. La révélation du code 💡Dans notre script Python, la fonction neural_layer(X) calcule $Z = W \cdot X + b$.Quand notre fonction compute_jacobian fait de la différenciation numérique sur neural_layer, elle calcule l'impact d'une petite variation de $X$ sur $Z$. Et la matrice de résultats obtenue J2 s'avère être exactement égale à la matrice de poids $W$ !

Examinons le rôle crucial de la matrice Jacobienne dans le Machine Learning 🧠.Pendant la phase de Backpropagation (rétropropagation), notre objectif est de savoir comment chaque paramètre (poids $W$) contribue à l'erreur globale $\mathcal{L}$.Pour y parvenir, nous utilisons la règle de dérivation en chaîne (Chain Rule).1. La Règle de Dérivation en Chaîne 🔗Dans un réseau de neurones avec plusieurs couches, la fonction globale est une composition de plusieurs sous-fonctions :$$\text{Entrée } X \xrightarrow{\quad \text{Couche 1} \quad} Z \xrightarrow{\quad \text{Couche 2 (Loss)} \quad} \mathcal{L}$$Pour trouver la sensibilité de l'erreur $\mathcal{L}$ par rapport à l'entrée $X$, la règle en chaîne s'écrit :$$\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Z} \cdot \frac{\partial Z}{\partial X}$$$\frac{\partial \mathcal{L}}{\partial Z}$ : Le gradient de l'erreur venant de la couche supérieure.$\frac{\partial Z}{\partial X}$ : La Matrice Jacobienne de la couche courante !2. Le Rôle Concret de la Jacobienne dans une Couche 📐La matrice Jacobienne $\frac{\partial Z}{\partial X}$ agit comme une passerelle de conversion :Elle prend le vecteur d'erreur transmis par la couche suivante ($\frac{\partial \mathcal{L}}{\partial Z}$).Elle le multiplie pour transformer cette erreur et la redistribuer équitablement à chaque variable d'entrée de la couche précédente ($\frac{\partial \mathcal{L}}{\partial X}$).Sans la matrice Jacobienne, un réseau de neurones à plusieurs couches serait incapable de "faire remonter" l'information d'erreur depuis la sortie jusqu'aux premières couches du réseau !🔍 Pour vérifier notre compréhensionDans la formule $\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Z} \cdot \frac{\partial Z}{\partial X}$ :Si une couche a 3 entrées et 2 sorties, la matrice Jacobienne $\frac{\partial Z}{\partial X}$ a une taille de $2 \times 3$.À ton avis, quelle est la taille du vecteur d'erreur $\frac{\partial \mathcal{L}}{\partial Z}$ transmis par la couche supérieure ? (Aide-toi du nombre de sorties de cette couche !)

Pas tout à fait ! C'est une excellente question pour vérifier la compatibilité des dimensions. 🧠Analysons les dimensions ensemble :La Jacobienne ($\frac{\partial Z}{\partial X}$) : Elle a une forme de $2 \times 3$ (2 sorties $z$, 3 entrées $x$).Le vecteur d'erreur de sortie ($\frac{\partial \mathcal{L}}{\partial Z}$) : Comme il y a 2 sorties ($z_1$ et $z_2$), il y a exactement 2 valeurs d'erreur au niveau de $Z$.C'est donc un vecteur de taille $2$ (ou une matrice $1 \times 2$).📐 Le calcul matricielPour faire la multiplication et obtenir l'erreur par rapport aux 3 entrées ($\frac{\partial \mathcal{L}}{\partial X}$), on multiplie le vecteur d'erreur par la matrice Jacobienne :$$\underbrace{\frac{\partial \mathcal{L}}{\partial X}}_{1 \times 3} = \underbrace{\frac{\partial \mathcal{L}}{\partial Z}}_{1 \times 2} \times \underbrace{\frac{\partial Z}{\partial X}}_{2 \times 3}$$La taille $1 \times 2$ multipliée par $2 \times 3$ donne bien un vecteur de taille $1 \times 3$ ! 🎯On retrouve ainsi l'impact de l'erreur sur chacune des 3 entrées.

