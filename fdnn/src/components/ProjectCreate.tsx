import Button from "@mui/material/Button";
import * as React from "react";
import TextField from "@mui/material/TextField";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import {
    Backdrop,
    CircularProgress,
    FormControl,
    InputLabel,
    MenuItem,
    Select,
    SelectChangeEvent,
    Typography,
    type DialogProps,
} from "@mui/material";
import { IDatasetState } from "./DatasetPage";
import { useNavigate } from "react-router-dom";
import { ProjectCreateLabelSplit } from "./ProjectCreateLabelSplit";

interface ICreateProjectRequest {
    name: string;
    dataset: string;
    image_width: number;
    image_height: number;
    label_split: { [key: string]: number };
    train_split: number;
    max_images?: number | null;
    seed?: number | null;
}

export function ProjectCreate() {
    const [open, setOpen] = React.useState(false);
    const navigate = useNavigate();
    const [isLoading, setIsLoading] = React.useState<boolean>(false);
    const [datasetState, setDatasetState] = React.useState<IDatasetState>({
        datasets: [],
    });
    const [formState, setFormState] = React.useState<ICreateProjectRequest>({
        name: "",
        dataset: "",
        image_width: 224,
        image_height: 224,
        label_split: {},
        train_split: 0.7,
        max_images: null,
        seed: null,
    });

    const handleDatasetChange = (event: SelectChangeEvent) => {
        const name = event.target.value as string;
        const selectedDataset = datasetState.datasets.find((h) => h.name === name);

        if (selectedDataset) {
            let labels = Object.keys(selectedDataset.labels);
            if (labels) {
                let defaultSplit = 1.0 / labels.length;
                let labelSplit: { [key: string]: number } = {};

                for (let i = 0; i < labels.length; i++) {
                    labelSplit[labels[i]] = defaultSplit;
                }

                setFormState({
                    ...formState,
                    dataset: event.target.value as string,
                    label_split: labelSplit,
                });
            }
        }
    };

    const setName = (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        setFormState({
            ...formState,
            name: event.target.value as string,
        });
    };

    const setWidth = (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        setFormState({
            ...formState,
            image_width: Number(event.target.value as string),
        });
    };

    const setHeight = (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        setFormState({
            ...formState,
            image_height: Number(event.target.value as string),
        });
    };

    const setTrainSplit = (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        setFormState({
            ...formState,
            train_split: Number(event.target.value as string),
        });
    };

    const setMaxImages = (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        setFormState({
            ...formState,
            max_images: Number(event.target.value as string),
        });
    };

    const setSeed = (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        setFormState({
            ...formState,
            seed: Number(event.target.value as string),
        });
    };

    const handleClose: DialogProps["onClose"] = (_event, reason) => {
        if (reason && reason === "backdropClick") return;
        setOpen(false);
    };

    const forceClose = () => setOpen(false);

    const handleClickOpen = () => {
        setIsLoading(true);
        fetch("/api/dataset/").then((res) =>
            res
                .json()
                .then((data: IDatasetState) => {
                    setDatasetState(data);
                })
                .then(() => {
                    setOpen(true);
                    setIsLoading(false);
                })
        );
    };

    const createProject = (name: string, description: string) => {};

    const setLabelSplit = (key: string, value: number) => {
        let formStateClone = JSON.parse(JSON.stringify(formState));

        formStateClone.label_split[key] = value;

        setFormState(formStateClone);
    };

    return (
        <div>
            <Backdrop sx={{ color: "#fff", zIndex: (theme) => theme.zIndex.drawer + 1 }} open={isLoading}>
                <Typography gutterBottom variant="body1">
                    This might take a minute.
                </Typography>
                <CircularProgress color="inherit" />
            </Backdrop>
            <Button onClick={handleClickOpen} variant="contained">
                Create Project
            </Button>
            <Dialog open={open} onClose={handleClose}>
                <DialogTitle>Create Project</DialogTitle>
                <DialogContent>
                    <TextField
                        fullWidth
                        style={{ marginTop: 10 }}
                        label="Name"
                        variant="outlined"
                        value={formState.name}
                        onChange={setName}
                    />
                    <FormControl fullWidth style={{ marginTop: 15 }}>
                        <InputLabel id="demo-simple-select-label">Dataset</InputLabel>
                        <Select
                            labelId="demo-simple-select-label"
                            id="demo-simple-select"
                            value={formState.dataset}
                            label="Dataset"
                            onChange={handleDatasetChange}
                        >
                            <MenuItem value={""}>None</MenuItem>
                            {datasetState.datasets.map((dataset) => (
                                <MenuItem key={dataset.name} value={dataset.name}>
                                    {dataset.name}
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                    <TextField
                        fullWidth
                        disabled={formState.dataset === ""}
                        style={{ marginTop: 15 }}
                        label="Images Resize Width"
                        variant="outlined"
                        value={formState.image_width}
                        onChange={setWidth}
                    />
                    <TextField
                        fullWidth
                        disabled={formState.dataset === ""}
                        style={{ marginTop: 15 }}
                        label="Images Resize Height"
                        variant="outlined"
                        value={formState.image_height}
                        onChange={setHeight}
                    />
                    <TextField
                        fullWidth
                        disabled={formState.dataset === ""}
                        style={{ marginTop: 15 }}
                        label="Train Split"
                        variant="outlined"
                        value={formState.train_split}
                        onChange={setTrainSplit}
                    />
                    <TextField
                        fullWidth
                        disabled={formState.dataset === ""}
                        style={{ marginTop: 15 }}
                        label="Max Images"
                        variant="outlined"
                        value={formState.max_images}
                        onChange={setMaxImages}
                    />
                    <TextField
                        fullWidth
                        disabled={formState.dataset === ""}
                        style={{ marginTop: 15 }}
                        label="Seed"
                        variant="outlined"
                        value={formState.seed}
                        onChange={setSeed}
                    />
                    <ProjectCreateLabelSplit label_splits={formState.label_split} setLabel={setLabelSplit} />
                </DialogContent>
                <DialogActions>
                    <Button onClick={forceClose}>Cancel</Button>
                    <Button onClick={createProject}>Create</Button>
                </DialogActions>
            </Dialog>
        </div>
    );
}
