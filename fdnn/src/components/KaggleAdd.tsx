import { DialogContent, DialogTitle } from "@mui/material";
import Button from "@mui/material/Button";
import Dialog, { DialogProps } from "@mui/material/Dialog";
import * as React from "react";
import KaggleAddWizard from "./KaggleAddWizard";
import { IDataset } from "./DatasetPage";

interface KaggleAddProps {
    dataset: IDataset;
    refresh: () => void;
}

export function KaggleAdd(props: KaggleAddProps) {
    const [open, setOpen] = React.useState(false);

    const handleClose: DialogProps["onClose"] = (_event, reason) => {
        if (reason && reason === "backdropClick") return;
        setOpen(false);
    };

    const forceClose = () => setOpen(false);

    const handleClickOpen = () => {
        setOpen(true);
    };

    return (
        <div>
            <Button onClick={handleClickOpen} variant="contained">
                Add Kaggle Dataset
            </Button>
            <Dialog open={open} onClose={handleClose} fullWidth={true} maxWidth="xl">
                <DialogTitle>Add Kaggle Dataset</DialogTitle>
                <DialogContent>
                    <KaggleAddWizard
                        forceClosed={forceClose}
                        refreshData={props.refresh}
                        datasetName={props.dataset.name}
                    />
                </DialogContent>
            </Dialog>
        </div>
    );
}
