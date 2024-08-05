import { Link } from "react-router-dom";
import { SelectedKaggleDataset } from "./KaggleAddWizard";
import {
    Alert,
    Box,
    Button,
    FormControl,
    InputLabel,
    MenuItem,
    Select,
    SelectChangeEvent,
    Typography,
} from "@mui/material";

interface UniqueValuesResponse {
    values: Array<string>;
}
interface KaggleColumnSelectionProps {
    dataset: SelectedKaggleDataset;
    next: () => void;
    back: () => void;
    cancel: () => void;
    csv: string;
    columns: Array<string>;
    imageColumn: string;
    labelColumn: string;
    folderColumn: string;
    setImageColumn: (imageColumn: string) => void;
    setLabelColumn: (labelColumn: string) => void;
    setFolderColumn: (folderColumn: string) => void;
    setAllLabels: (labels: Array<string>) => void;
    setIsLoading: (isLoading: boolean) => void;
}

export function KaggleColumnSelection(props: KaggleColumnSelectionProps) {
    const handleImageColumnChange = (event: SelectChangeEvent) => {
        props.setImageColumn(event.target.value as string);
    };

    const handleLabelColumnChange = (event: SelectChangeEvent) => {
        props.setLabelColumn(event.target.value as string);
    };

    const handleFolderColumnChange = (event: SelectChangeEvent) => {
        props.setFolderColumn(event.target.value as string);
    };

    const onNext = () => {
        props.setIsLoading(true);
        fetch(
            `/api/dataset/unique?dataset_identifier=${props.dataset.identifier}&path_to_metadata=${props.csv}&column=${props.labelColumn}`
        )
            .then(function (res) {
                return res.json();
            })
            .then(function (response: UniqueValuesResponse) {
                props.setAllLabels(response.values);
                props.next();
                props.setIsLoading(false);
            });
    };

    return (
        <div>
            <Alert severity="success">
                <Typography gutterBottom variant="h6">
                    {props.dataset.identifier} - <Link to={props.dataset.url}>{props.dataset.url}</Link>
                </Typography>
            </Alert>
            <FormControl fullWidth style={{ marginTop: 15 }}>
                <InputLabel id="demo-simple-select-label">Image Column</InputLabel>
                <Select
                    labelId="demo-simple-select-label"
                    id="demo-simple-select"
                    value={props.imageColumn}
                    label="Image Column"
                    onChange={handleImageColumnChange}
                >
                    <MenuItem value={""}>None</MenuItem>
                    {props.columns.map((image) => (
                        <MenuItem key={image} value={image}>
                            {image}
                        </MenuItem>
                    ))}
                </Select>
            </FormControl>
            <FormControl fullWidth style={{ marginTop: 10 }}>
                <InputLabel id="demo-simple-select-label">Label Column</InputLabel>
                <Select
                    labelId="demo-simple-select-label"
                    id="demo-simple-select"
                    value={props.labelColumn}
                    label="Label Column"
                    onChange={handleLabelColumnChange}
                >
                    <MenuItem value={""}>None</MenuItem>
                    {props.columns.map((label) => (
                        <MenuItem key={label} value={label}>
                            {label}
                        </MenuItem>
                    ))}
                </Select>
            </FormControl>
            <FormControl fullWidth style={{ marginTop: 10 }}>
                <InputLabel id="demo-simple-select-label">Folder Column</InputLabel>
                <Select
                    labelId="demo-simple-select-label"
                    id="demo-simple-select"
                    value={props.folderColumn}
                    label="Folder Column"
                    onChange={handleFolderColumnChange}
                >
                    <MenuItem value={""}>None</MenuItem>
                    {props.columns.map((folder) => (
                        <MenuItem key={folder} value={folder}>
                            {folder}
                        </MenuItem>
                    ))}
                </Select>
            </FormControl>
            <Box sx={{ display: "flex", flexDirection: "row", pt: 2 }}>
                <Button color="inherit" onClick={props.back} sx={{ mr: 1 }}>
                    Back
                </Button>
                <Button disabled={props.imageColumn === "" || props.labelColumn === ""} onClick={onNext}>
                    Next
                </Button>
                <Box sx={{ flex: "1 1 auto" }} />
                <Button onClick={props.cancel}>Cancel</Button>
            </Box>
        </div>
    );
}
