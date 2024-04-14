import Button from "@mui/material/Button";
import * as React from "react";
import TextField from "@mui/material/TextField";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import type { DialogProps } from "@mui/material";

export interface IDatasetCreateProps {
    refreshDatasets: () => void;
}

export function DatasetCreate(props: IDatasetCreateProps) {
    const [open, setOpen] = React.useState(false);

    const handleClose: DialogProps["onClose"] = (_event, reason) => {
        if (reason && reason === "backdropClick") return;
        setOpen(false);
    };

    const forceClose = () => setOpen(false);

    const handleClickOpen = () => {
        setOpen(true);
    };

    const createDataset = (name: string, description: string) => {
        fetch("/api/dataset/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                name: name,
                description: description,
            }),
        })
            .then(function (res) {
                return res.json();
            })
            .then(function () {
                props.refreshDatasets();
                setOpen(false);
            });
    };

    return (
        <div>
            <Button onClick={handleClickOpen} variant="contained">
                Create Dataset
            </Button>
            <Dialog
                open={open}
                onClose={handleClose}
                PaperProps={{
                    component: "form",
                    onSubmit: (event: React.FormEvent<HTMLFormElement>) => {
                        event.preventDefault();
                        const formData = new FormData(event.currentTarget);
                        const formJson = Object.fromEntries((formData as any).entries());
                        const name = formJson.name;
                        const description = formJson.name;
                        createDataset(name, description);
                    },
                }}
            >
                <DialogTitle>Create Dataset</DialogTitle>
                <DialogContent>
                    <TextField
                        autoFocus
                        required
                        margin="dense"
                        id="name"
                        name="name"
                        label="Name"
                        type="text"
                        fullWidth
                        variant="standard"
                    />
                    <TextField
                        autoFocus
                        required
                        margin="dense"
                        id="description"
                        name="description"
                        label="Description"
                        type="text"
                        fullWidth
                        variant="standard"
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={forceClose}>Cancel</Button>
                    <Button type="submit">Create</Button>
                </DialogActions>
            </Dialog>
        </div>
    );
}
