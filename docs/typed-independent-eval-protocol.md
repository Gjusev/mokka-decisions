# Protocolo de evaluación independiente — decisiones tipadas (frozen v1.0)

Fecha de congelación: 2026-10-08, ANTES de abrir cualquier evaluación nueva y
antes de seleccionar finalista de E15. Motivación: los desgloses por dominio de
E13/E14 son particiones del test histórico ya consultado — congelar un desglose
no crea independencia (corrección de la revisión 2026-10-08). Este documento
fija qué será una evaluación genuinamente independiente y cómo se construye.

## Objetivo

Medir generalización a casos y familias NO vistos en ningún split consultado
(train/dev/cal ni el test histórico), sin construir el conjunto a partir de
errores del test histórico y sin usar sus resultados para elegir
hiperparámetros o fabricar ejemplos.

## Fuentes admitidas (por orden de independencia)

1. **Generador propio con etiquetas verificables por regla** (preferido).
   Casos nuevos construidos por plantillas/familias distintas de las del
   dataset original, con etiqueta derivada del propio generador (verificación
   programática, procedencia = código + seed pineado). Declaración de alcance
   obligatoria: mide generalización a esquemas nuevos, no distribución real.
2. **Particiones del dataset upstream jamás consultadas por este proyecto**,
   solo si se documenta su disyunción por hash de case id/familia frente a
   TODOS los splits ya usados (train, dev, cal, all/test y los 4 tests de
   dominio) y se declara la posibilidad de solape con el entrenamiento de
   terceros (p. ej. Julia) — ese solape no invalida la medición de
   generalización de NUESTRO modelo, pero se declara.
3. **Nunca**: el test histórico, sus errores, sus ids ni derivados suyos.

## Reglas de construcción

- Unidades de agrupación: caso original; familia/plantilla como estrato. Los
  splits se hacen por caso ANTES de expandir preguntas; una familia no puede
  aparecer a la vez en cualquier split de entrenamiento y aquí.
- Etiquetas: verificables por regla (generador) o doble anotación con
  arbitraje registrado. Se conserva `gold.probabilities` si existe, separada
  de la etiqueta dura.
- Congelación previa: lista de ids + sha256 del archivo + código generador
  pineado se registran en un manifiesto ANTES de que cualquier finalista lo
  ejecute. Sin re-generación tras ver resultados.
- Exposición: cualquier consulta de resultados agregados sobre este conjunto
  se registra con fecha y decisión que motivó.

## Protocolo de ejecución (idéntico al adaptador frozen v1.0)

- `json.dumps(state, ensure_ascii=False)` como serialización; orden de
  criteria preservado; sin `none` añadido; `max_length` 512 (A0) / máx del
  modelo (A1); temperatura 1.0; batch declarado por corrida.
- Métricas: acierto global y por tipo (choice/noul/score), NLL, KL frente a
  `gold.probabilities` cuando exista, Brier multiclase, MAE ordinal (valor
  esperado). Comparadores relevantes bajo el mismo contrato (Julia oficial
  incluido si su licencia lo permite).
- Inferencia agrupada: intervalos por bootstrap de casos (grupos = caso
  original, 5 preguntas/caso en el benchmark original; aquí grupos = familia
  si el generador lo permite, si no caso). Diferencias pareadas por caso
  (McNemar por pregunta como referencia secundaria, nunca único criterio).
- Réplicas: si una comparación decide continuación, ≥3 semillas emparejadas
  por receta y cuota disponible; se reporta media y dispersión por receta.

## Análisis pre-registrado

- Criterio de éxito de fase (sin cambios): ≥+3 pp acierto vs Julia-1 o ≥30 %
  p95 con ≤1 pp de pérdida, medidos bajo condiciones de servicio idénticas
  (mismo batch, mismo hardware, mismo backend).
- Toda decisión de continuación se toma sobre dev o sobre este conjunto
  independiente; el test histórico queda como referencia histórica declarada.

## Estado

- Diseño congelado. Construcción del conjunto: tras el resultado de E15
  (señal → réplicas primero; sin señal → el conjunto sirve a la siguiente
  comparación). El manifiesto se añadirá a `data/independent-v1/` con hashes
  antes del primer uso.
