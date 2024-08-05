import { SelectedKaggleDataset } from "./KaggleAddWizard";
import {
    Alert,
    Box,
    Button,
    Checkbox,
    FormControl,
    FormControlLabel,
    InputLabel,
    MenuItem,
    Paper,
    Select,
    SelectChangeEvent,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    TextField,
    Typography,
} from "@mui/material";
import * as React from "react";
import { Link } from "react-router-dom";

interface KaggleRelabelProps {
    dataset: SelectedKaggleDataset;
    next: () => void;
    back: () => void;
    cancel: () => void;
    allFolders: Array<string>;
    allLabels: Array<string>;
    setIsLoading: (isLoading: boolean) => void;
    setLabels: (labels: Array<RelabelItem>) => void;
}

export interface RelabelItem {
    label: string;
    folder: string;
    relabel: string;
    include: boolean;
}

export function KaggleRelabel(props: KaggleRelabelProps) {
    const [state, setState] = React.useState<Array<RelabelItem>>([]);
    const [currentStateItem, setCurrentStateItem] = React.useState<RelabelItem>({
        label: "",
        folder: "",
        relabel: "",
        include: true,
    });

    const handleLabelChange = (event: SelectChangeEvent) => {
        setCurrentStateItem({
            ...currentStateItem,
            label: event.target.value as string,
        });
    };

    const handleFolderChange = (event: SelectChangeEvent) => {
        setCurrentStateItem({
            ...currentStateItem,
            folder: event.target.value as string,
        });
    };

    const onNext = () => {
        props.setLabels(state);
    };

    const setReLabel = (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        setCurrentStateItem({
            ...currentStateItem,
            relabel: event.target.value as string,
        });
    };

    const addButtonClicked = () => {
        setState([...state, currentStateItem]);
        setCurrentStateItem({
            label: "",
            folder: "",
            relabel: "",
            include: true,
        });
    };

    const onIncludeChange = (_event: React.SyntheticEvent<Element, Event>, checked: boolean) => {
        setCurrentStateItem({
            ...currentStateItem,
            include: checked,
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
                <InputLabel id="demo-simple-select-label">Dataset Label</InputLabel>
                <Select
                    labelId="demo-simple-select-label"
                    id="demo-simple-select"
                    value={currentStateItem.label}
                    label="Dataset Label"
                    onChange={handleLabelChange}
                >
                    <MenuItem value={""}>None</MenuItem>
                    {props.allLabels.map((label) => (
                        <MenuItem key={label} value={label}>
                            {label}
                        </MenuItem>
                    ))}
                </Select>
            </FormControl>
            <FormControl fullWidth style={{ marginTop: 10 }}>
                <InputLabel id="demo-simple-select-label">Folder to images</InputLabel>
                <Select
                    labelId="demo-simple-select-label"
                    id="demo-simple-select"
                    value={currentStateItem.folder}
                    label="Folder to images"
                    onChange={handleFolderChange}
                >
                    <MenuItem value={""}>None</MenuItem>
                    {props.allFolders.map((folder) => (
                        <MenuItem key={folder} value={folder}>
                            {folder}
                        </MenuItem>
                    ))}
                </Select>
            </FormControl>
            <TextField
                fullWidth
                style={{ marginTop: 10 }}
                label="New Label"
                variant="outlined"
                value={currentStateItem.relabel}
                onChange={setReLabel}
            />
            <div>
                <FormControlLabel
                    value="Contains"
                    control={<Checkbox />}
                    label="Contains"
                    labelPlacement="start"
                    onChange={onIncludeChange}
                />
            </div>
            <Button variant="contained" onClick={addButtonClicked} sx={{ mr: 1, mt: 1, mb: 4 }}>
                Add Label
            </Button>

            <TableContainer sx={{ maxHeight: 300 }} component={Paper}>
                <Table stickyHeader size="small" sx={{ minWidth: 650 }} aria-label="simple table">
                    <TableHead>
                        <TableRow>
                            <TableCell>Label</TableCell>
                            <TableCell>Folder</TableCell>
                            <TableCell>New Label</TableCell>
                            <TableCell>Contains</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {state.map((row) => (
                            <TableRow key={row.relabel} sx={{ "&:last-child td, &:last-child th": { border: 0 } }}>
                                <TableCell component="th" scope="row">
                                    {row.label}
                                </TableCell>
                                <TableCell>{row.folder}</TableCell>
                                <TableCell>{row.relabel}</TableCell>
                                <TableCell>{row.include ? "true" : "false"}</TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>

            <Box sx={{ display: "flex", flexDirection: "row", pt: 2 }}>
                <Button color="inherit" onClick={props.back} sx={{ mr: 1 }}>
                    Back
                </Button>
                <Button disabled={state.length <= 1} onClick={onNext}>
                    Add Dataset
                </Button>
                <Box sx={{ flex: "1 1 auto" }} />
                <Button onClick={props.cancel}>Cancel</Button>
            </Box>
        </div>
    );
}
