# Descripción de Datos — Titanic

Este dataset proviene del clásico problema de predicción de supervivencia en el Titanic (1912).  
Cada fila corresponde a un pasajero y contiene las siguientes variables:

| Variable   | Descripción                                                                 |
|------------|-----------------------------------------------------------------------------|
| PassengerId| Identificador único del pasajero                                            |
| Survived   | Variable objetivo (0 = No sobrevivió, 1 = Sobrevivió)                       |
| Pclass     | Clase del ticket (1 = Primera, 2 = Segunda, 3 = Tercera)                   |
| Name       | Nombre completo del pasajero, incluye título social                        |
| Sex        | Sexo del pasajero (male, female)                                           |
| Age        | Edad en años (puede contener valores nulos)                                |
| SibSp      | Número de hermanos/as o cónyuge a bordo                                    |
| Parch      | Número de padres/madres o hijos a bordo                                    |
| Ticket     | Número del boleto (puede contener duplicados)                              |
| Fare       | Tarifa pagada por el pasajero (en libras de 1912)                          |
| Cabin      | Número de cabina (muy incompleto, muchos valores nulos)                    |
| Embarked   | Puerto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton)        |

📌 **Notas**:  
- A partir de estas variables se derivan nuevas features como *Title*, *FamilySize*, *FarePerPerson*, *IsAlone*, *HasCabin* y *Fare_log*.  
- Las variables categóricas (`Sex`, `Pclass`, `Embarked`, `Title`) fueron transformadas con One-Hot Encoding para el modelado.  
