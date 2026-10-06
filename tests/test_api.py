def test_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "API de prédiction d'attrition opérationnelle"
    }


def test_read_employee(client, employee_id):
    response = client.get(
        f"/employees/{employee_id}"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == employee_id


def test_employee_not_found(
    client,
    employee_id_inexistant,
):
    response = client.get(
        f"/employees/{employee_id_inexistant}"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Employé introuvable"
    }


def test_predict(
    client,
    payload_employee,
):
    response = client.post(
        "/predict",
        json=payload_employee,
    )

    assert response.status_code == 201

    data = response.json()

    assert data["employee_id"] == payload_employee["employee_id"]

    assert data["classe_predite"] in {
        "Oui",
        "Non",
    }

    assert 0.0 <= data["probabilite_non"] <= 1.0
    assert 0.0 <= data["probabilite_oui"] <= 1.0

    assert (
        abs(
            data["probabilite_non"]
            + data["probabilite_oui"]
            - 1.0
        )
        < 1e-9
    )


def test_predict_employee_not_found(
    client,
    payload_employee,
    employee_id_inexistant,
):
    payload = payload_employee.copy()

    payload["employee_id"] = employee_id_inexistant

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Employé introuvable"
    }