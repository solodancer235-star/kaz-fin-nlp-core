// CI/CD pipeline for kaz-fin-nlp-core (Assignment 5)
// CI : every commit -> lint, type check, security scan, unit tests + coverage gate,
//      Docker image build, container smoke test
// CD : main branch  -> push image to Docker Hub, deploy to Minikube (Kubernetes)

pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '20'))
        timeout(time: 30, unit: 'MINUTES')
    }

    // Webhook from GitHub on every push; SCM polling as a fallback when the
    // Jenkins server is not reachable from the internet (e.g. localhost).
    triggers {
        githubPush()
        pollSCM('H/2 * * * *')
    }

    environment {
        // Change to your Docker Hub namespace
        IMAGE_NAME   = 'solodancer235/kaz-fin-nlp-core'
        IMAGE_TAG    = "${env.BUILD_NUMBER}-${env.GIT_COMMIT?.take(7) ?: 'local'}"
        VENV         = "${env.WORKSPACE}/.venv"
        COVERAGE_MIN = '90'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                sh 'git log -1 --pretty="format:%h %an: %s"'
            }
        }

        stage('Set up Python') {
            steps {
                sh '''
                    python3 -m venv "$VENV"
                    . "$VENV/bin/activate"
                    pip install --quiet --upgrade pip
                    pip install --quiet -r requirements-dev.txt
                    pip install --quiet -e .
                '''
            }
        }

        stage('Static checks') {
            parallel {
                stage('Lint (black + flake8)') {
                    steps {
                        sh '''
                            . "$VENV/bin/activate"
                            black --check kaz_fin_nlp tests
                            flake8 kaz_fin_nlp tests --max-line-length=88 --extend-ignore=E203,W503
                        '''
                    }
                }
                stage('Type check (mypy)') {
                    steps {
                        sh '. "$VENV/bin/activate" && mypy kaz_fin_nlp'
                    }
                }
                stage('Security (bandit)') {
                    steps {
                        sh '. "$VENV/bin/activate" && bandit -r kaz_fin_nlp -ll'
                    }
                }
            }
        }

        stage('Unit tests') {
            steps {
                sh '''
                    . "$VENV/bin/activate"
                    mkdir -p reports
                    pytest --junitxml=reports/junit.xml \
                           --cov=kaz_fin_nlp --cov-report=xml:reports/coverage.xml \
                           --cov-report=term --cov-fail-under=$COVERAGE_MIN
                '''
            }
            post {
                always {
                    junit 'reports/junit.xml'
                    archiveArtifacts artifacts: 'reports/*.xml', allowEmptyArchive: true
                }
            }
        }

        stage('Build Docker image') {
            steps {
                sh '''
                    docker build \
                      --build-arg GIT_COMMIT=$GIT_COMMIT \
                      --build-arg BUILD_NUMBER=$BUILD_NUMBER \
                      -t $IMAGE_NAME:$IMAGE_TAG -t $IMAGE_NAME:latest .
                    docker image ls $IMAGE_NAME
                '''
            }
        }

        stage('Smoke test container') {
            steps {
                sh '''
                    docker rm -f kfn-smoke 2>/dev/null || true
                    docker run -d --name kfn-smoke $IMAGE_NAME:$IMAGE_TAG
                    for i in $(seq 1 20); do
                      if docker exec kfn-smoke python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"; then
                        echo "container healthy"; break
                      fi
                      sleep 1
                    done
                    docker exec kfn-smoke python -c "import urllib.request,json; r=urllib.request.Request('http://127.0.0.1:8000/analyze', data=json.dumps({'text':'KZTO net profit up 35%','source':'JENKINS_SMOKE'}).encode(), headers={'Content-Type':'application/json'}); print(urllib.request.urlopen(r).read().decode())"
                '''
            }
            post {
                always { sh 'docker rm -f kfn-smoke 2>/dev/null || true' }
            }
        }

        stage('Push to Docker Hub') {
            when { branch 'main' }
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-creds',
                                                  usernameVariable: 'DH_USER',
                                                  passwordVariable: 'DH_TOKEN')]) {
                    sh '''
                        echo "$DH_TOKEN" | docker login -u "$DH_USER" --password-stdin
                        docker push $IMAGE_NAME:$IMAGE_TAG
                        docker push $IMAGE_NAME:latest
                        docker logout
                    '''
                }
            }
        }

        stage('Deploy to Minikube') {
            when { branch 'main' }
            steps {
                withKubeConfig([credentialsId: 'minikube-kubeconfig']) {
                    sh '''
                        kubectl apply -f k8s/
                        kubectl set image deployment/kaz-fin-nlp api=$IMAGE_NAME:$IMAGE_TAG
                        kubectl rollout status deployment/kaz-fin-nlp --timeout=120s
                        kubectl get pods -l app=kaz-fin-nlp -o wide
                    '''
                }
            }
        }
    }

    post {
        success { echo "Pipeline succeeded: ${IMAGE_NAME}:${IMAGE_TAG}" }
        failure { echo 'Pipeline failed - see the stage logs above.' }
        always  { sh 'docker image prune -f || true' }
    }
}
