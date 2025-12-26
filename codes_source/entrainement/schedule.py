 
"""
Gestion du scheduling (warmup) du coefficient lambda.

Dans Fader Networks, on n'applique pas l'adversarial 
dès le début.
On commence par apprendre à reconstruire (autoencoder), 
puis on augmente progressivement 
l'influence du discriminateur latent.

 
"""


def lambda_schedule(step: int, lambda_max: float, warmup_steps: int) -> float:
    """
    Calcul de lambda(t) avec warmup linéaire.

    Paramètres
    ----------
    step : int
        "compteur" d'itérations global (nombre d'images vues ou nombre de steps).
    lambda_max : float
        valeur maximale de lambda (ex: 1e-4).
    warmup_steps : int
        nombre d'itérations nécessaires pour atteindre lambda_max.
        - si 0 ou négatif : lambda constant = lambda_max.

    Retour
    ------
    float
        lambda(t) dans [0, lambda_max].

    Exemple
    -------
    - warmup_steps = 500000
    - step = 0          -> lambda = 0
    - step = 250000     -> lambda = 0.5 * lambda_max
    - step >= 500000    -> lambda = lambda_max
    """
    if warmup_steps is None or warmup_steps <= 0:
        return float(lambda_max)

    ratio = step / float(warmup_steps)
    if ratio > 1.0:
        ratio = 1.0
    elif ratio < 0.0:
        ratio = 0.0

    return float(lambda_max) * ratio
