import * as React from "react";
import Box from "@mui/material/Box";
import Stepper from "@mui/material/Stepper";
import Step from "@mui/material/Step";
import StepLabel from "@mui/material/StepLabel";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import { KaggleSearch } from "./KaggleSearch";
import { Backdrop, CircularProgress } from "@mui/material";
import { KaggleMetadataSelection } from "./KaggleMetadataSelection";
import { KaggleColumnSelection } from "./KaggleColumnSelection";
import { KaggleRelabel, RelabelItem } from "./KaggleRelabel";

const steps = ["Select Kaggle Dataset", "Set CSV", "Set Columns", "Re-label Data"];

export interface KaggleDatasetMetadata {
    csv: Array<string>;
    subfolders: Array<string>;
}

export interface SelectedKaggleDataset {
    identifier: string;
    url: string;
    metadata: KaggleDatasetMetadata;
}

interface KaggleAddWizardProps {
    forceClosed: () => void;
    refreshData: () => void;
    datasetName: string;
}

export default function KaggleAddWizard(props: KaggleAddWizardProps) {
    const { forceClosed } = props;
    const [activeStep, setActiveStep] = React.useState(0);
    const [csv, setCsv] = React.useState("");
    const [imageColumn, setImageColumn] = React.useState("");
    const [labelColumn, setLabelColumn] = React.useState("");
    const [folderColumn, setFolderColumn] = React.useState("");
    const [isLoading, setIsLoading] = React.useState<boolean>(false);
    const [columns, setColumns] = React.useState<Array<string>>([]);
    const [allLabels, setAllLabels] = React.useState<Array<string>>([]);
    const [selectedKaggleDataset, setSelectedKaggleDataset] = React.useState<SelectedKaggleDataset>({
        identifier: "",
        url: "",
        metadata: {
            csv: [],
            subfolders: [],
        },
    });

    const handleNext = () => {
        setActiveStep((prevActiveStep) => prevActiveStep + 1);
    };

    const handleBack = () => {
        setActiveStep((prevActiveStep) => prevActiveStep - 1);
    };

    const handleReset = () => {
        setActiveStep(0);
    };

    const setSelectedDatasets = (identifier: string, url: string) => {
        setIsLoading(true);

        fetch("/api/dataset/download", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                dataset_identifier: identifier,
            }),
        })
            .then(function (res) {
                return res.json();
            })
            .then(function (response: KaggleDatasetMetadata) {
                setSelectedKaggleDataset({
                    identifier: identifier,
                    url: url,
                    metadata: response,
                });
                setActiveStep((prevActiveStep) => prevActiveStep + 1);
                setIsLoading(false);
            });
    };

    const addDataset = (relabelItems: Array<RelabelItem>) => {
        setIsLoading(true);

        const data = {
            name: props.datasetName,
            dataset_identifier: selectedKaggleDataset.identifier,
            dataset_kaggle_url: selectedKaggleDataset.url,
            path_to_metadata: csv,
            label_column: labelColumn,
            labels: relabelItems.map((relabel) => {
                return {
                    label: relabel.relabel,
                    folder: relabel.folder,
                    alias: relabel.label,
                    included: relabel.include,
                };
            }),
            image_column: imageColumn,
            folder_column: folderColumn,
            folder_to_lower: true,
        };

        fetch("/api/dataset/add", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(data),
        }).then(function () {
            props.refreshData();
            forceClosed();
            setIsLoading(false);
        });
    };

    const getContent = () => {
        if (activeStep === steps.length) {
            return (
                <React.Fragment>
                    <Typography sx={{ mt: 2, mb: 1 }}>All steps completed - you&apos;re finished</Typography>
                    <Box sx={{ display: "flex", flexDirection: "row", pt: 2 }}>
                        <Box sx={{ flex: "1 1 auto" }} />
                        <Button onClick={handleReset}>Reset</Button>
                    </Box>
                </React.Fragment>
            );
        } else if (activeStep === 0) {
            return <KaggleSearch setSelectedDatasets={setSelectedDatasets} />;
        } else if (activeStep === 1) {
            return (
                <KaggleMetadataSelection
                    dataset={selectedKaggleDataset}
                    next={handleNext}
                    back={handleBack}
                    cancel={forceClosed}
                    setCsv={setCsv}
                    csv={csv}
                    setIsLoading={setIsLoading}
                    setColumns={setColumns}
                />
            );
        } else if (activeStep === 2) {
            return (
                <KaggleColumnSelection
                    dataset={selectedKaggleDataset}
                    next={handleNext}
                    back={handleBack}
                    cancel={forceClosed}
                    csv={csv}
                    columns={columns}
                    imageColumn={imageColumn}
                    labelColumn={labelColumn}
                    folderColumn={folderColumn}
                    setImageColumn={setImageColumn}
                    setLabelColumn={setLabelColumn}
                    setFolderColumn={setFolderColumn}
                    setAllLabels={setAllLabels}
                    setIsLoading={setIsLoading}
                />
            );
        } else {
            return (
                <KaggleRelabel
                    dataset={selectedKaggleDataset}
                    next={handleNext}
                    back={handleBack}
                    cancel={forceClosed}
                    setIsLoading={setIsLoading}
                    allFolders={selectedKaggleDataset.metadata.subfolders}
                    allLabels={allLabels}
                    setLabels={addDataset}
                />
            );
        }
    };

    return (
        <Box sx={{ width: "100%" }}>
            <Backdrop sx={{ color: "#fff", zIndex: (theme) => theme.zIndex.drawer + 1 }} open={isLoading}>
                <Typography gutterBottom variant="body1">
                    This might take a minute.
                </Typography>
                <CircularProgress color="inherit" />
            </Backdrop>
            <Stepper activeStep={activeStep}>
                {steps.map((label) => {
                    const stepProps: { completed?: boolean } = {};
                    const labelProps: {
                        optional?: React.ReactNode;
                    } = {};
                    return (
                        <Step key={label} {...stepProps}>
                            <StepLabel {...labelProps}>{label}</StepLabel>
                        </Step>
                    );
                })}
            </Stepper>
            {getContent()}
        </Box>
    );
}
