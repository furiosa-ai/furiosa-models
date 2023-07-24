# Quit if error occurs
set -e

NAMESPACE="ci-furiosa-models"

echo "Namespace: $NAMESPACE"

# Change cwd to script path
cd "$(dirname "$(realpath "$0")")";

# Check kubectl command exists
if ! command -v kubectl &> /dev/null
then
    echo "kubectl command not found"
    exit 1
fi

# Check tkn command exists
if ! command -v tkn &> /dev/null
then
    echo "tkn command not found"
    exit 1
fi

# Apply tekton hub's scripts (reinstalls if install command fails)
tkn hub install task git-clone --namespace $NAMESPACE \
    || tkn hub reinstall task git-clone --namespace $NAMESPACE

tkn hub install task github-set-status --namespace $NAMESPACE \
    || tkn hub reinstall task github-set-status --namespace $NAMESPACE

# Apply all files in cwd
kubectl apply --namespace $NAMESPACE --filename .
