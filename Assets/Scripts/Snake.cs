using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

[RequireComponent(typeof(BoxCollider2D))]
public class SnakeAgent : Agent
{
    public Transform segmentPrefab;
    public Transform food;  // Referencia al objeto de comida.
    public Vector2Int direction = Vector2Int.right;
    public float speed = 20f;
    public int initialSize = 4;

    private readonly List<Transform> segments = new List<Transform>();
    private Vector2Int input;
    private Vector3 previousFoodDistance;

    public override void OnEpisodeBegin()
    {
        ResetState();
        MoveFood(); // Reubicar comida.
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Posición de la cabeza y de la comida.
        sensor.AddObservation(transform.position);
        sensor.AddObservation(food.position);
        sensor.AddObservation(direction);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        int action = actions.DiscreteActions[0];

        switch (action)
        {
            case 0: input = Vector2Int.up; break;
            case 1: input = Vector2Int.down; break;
            case 2: input = Vector2Int.left; break;
            case 3: input = Vector2Int.right; break;
        }

        MoveSnake();

        // Recompensas y penalizaciones según la distancia a la comida.
        Vector3 currentFoodDistance = food.position - transform.position;

        if (currentFoodDistance.magnitude < previousFoodDistance.magnitude)
        {
            AddReward(0.1f); // Premio por acercarse.
        }
        else
        {
            AddReward(-0.1f); // Penalización por alejarse.
        }

        previousFoodDistance = currentFoodDistance;
    }

    private void MoveSnake()
    {
        if (input != Vector2Int.zero) direction = input;

        for (int i = segments.Count - 1; i > 0; i--)
        {
            segments[i].position = segments[i - 1].position;
        }

        transform.position = new Vector2(
            Mathf.RoundToInt(transform.position.x) + direction.x,
            Mathf.RoundToInt(transform.position.y) + direction.y
        );

        CheckCollision();
    }

    private void CheckCollision()
    {
        // Colisión con la comida.
        if (Mathf.RoundToInt(transform.position.x) == Mathf.RoundToInt(food.position.x) &&
            Mathf.RoundToInt(transform.position.y) == Mathf.RoundToInt(food.position.y))
        {
            AddReward(1.0f); // Recompensa grande por comer la comida.
            Grow();
            MoveFood();
        }

        // Colisión consigo misma.
        for (int i = 1; i < segments.Count; i++)
        {
            if (segments[i].position == transform.position)
            {
                AddReward(-1.0f); // Penalización fuerte por morir.
                EndEpisode();
            }
        }
    }

    public void Grow()
    {
        Transform segment = Instantiate(segmentPrefab);
        segment.position = segments[segments.Count - 1].position;
        segments.Add(segment);
    }

    public void ResetState()
    {
        direction = Vector2Int.right;
        transform.position = Vector3.zero;
        previousFoodDistance = food.position - transform.position;

        foreach (Transform segment in segments)
        {
            if (segment != transform)
                Destroy(segment.gameObject);
        }

        segments.Clear();
        segments.Add(transform);

        for (int i = 0; i < initialSize - 1; i++)
        {
            Grow();
        }
    }

    private void MoveFood()
    {
        food.position = new Vector2(Random.Range(-10, 10), Random.Range(-10, 10));
    }

        public bool Occupies(int x, int y)
    {
        foreach (Transform segment in segments)
        {
            if (Mathf.RoundToInt(segment.position.x) == x &&
                Mathf.RoundToInt(segment.position.y) == y) {
                return true;
            }
        }

        return false;
    }
}